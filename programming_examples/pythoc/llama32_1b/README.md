# LLAMA-3.2-1B BF16 Inference on AMD NPU2 — PythoC + placed IRON

End-to-end Llama-3.2-1B inference on AMD NPU2 (Strix Halo, aie2p), built
incrementally on top of the MLIR-AIR reference at
[`mlir-air-llama_awq_impl/programming_examples/llama32_1b_aie`][air-src].
Each kernel is replaced with a PythoC `@aie_kernel` function and each
AIR multi-launch is replaced with a placed-IRON (`aie/aiex`-dialect)
Python builder; the cached AIR-emitted MLIR serves as the substrate
behind a feature flag until each builder lands.

[air-src]: ../../../../mlir-air-llama_awq_impl/programming_examples/llama32_1b_aie

## Status

### Kernels — PythoC `@aie_kernel` (all ship as `.o` linked by aiecc)

| Kernel | Where used | Source | Built `.o` |
|---|---|---|---|
| RMSNorm K=2048 (decode) | rms_gemv_rope | `kernels/rms_norm.py` | `rms_norm_2048_bf16.o` ✓ |
| silu_and_mul (SwiGLU) | o_gemv_ffn, o_ffn | `kernels/silu_and_mul.py` | `silu_and_mul_bf16.o` ✓ |
| rope (half-split) | rms_gemv_rope, rms_gemms_rope | `kernels/rope.py` | `rope_pythoc.o` ✓ |
| matvec (BF16 GEMV K=2048) | rms_gemv_rope, o_gemv_ffn, lm_head_gemv | `kernels/matvec.py` | `mv_pythoc.o` ✓ |
| matvec (BF16 GEMV K=8192) | o_gemv_ffn (FFN down) | `kernels/matvec_k8192.py` | `mv_k8192_pythoc.o` ✓ |
| Flash-attention primitives (19) | flash_attn | `kernels/attn.py` | `attn_pythoc.o` ✓ |
| BF16 GEMM (prefill matmuls) | rms_gemms_rope (v_matmul), o_ffn | `kernels/bf16_gemm.py` | `bf16_gemm_pythoc_M16_N8_K4_AT_bf16out_*.o` ✓ |
| AWQ uint4 GEMV | (AWQ path) | ☐ Phase 6 deferred | — |

`reference_o/` is empty — no `.cc`-built `.o` left in the project.

### Placed-IRON builders — 5 of 6 done

| Builder | Phase | Used by | Status |
|---|---|---|---|
| `builders/lm_head_gemv.py` | decode (final logits) | `llama32_1b_decode.py` | ✓ Phase 4.1 |
| `builders/rms_gemv_rope.py` | decode (RMSNorm + QKV GEMV + RoPE) | per-layer decode | ✓ Phase 4.3 |
| `builders/o_gemv_ffn.py` | decode (O + FFN) | per-layer decode | ✓ Phase 4.4 |
| `builders/flash_attn.py` | prefill flash attention | `llama32_1b_prefill.py` | ✓ Phase 4.2 |
| `builders/rms_gemms_rope.py` | prefill (RMSNorm + QKV GEMM + RoPE) | per-layer prefill | ✓ Phase 4.5 |
| `builders/o_ffn.py` | prefill (O + FFN with GEMMs) | per-layer prefill | ◐ Phase 4.6 (5 of 9 devices placed; 4 GEMM devices deferred via cached-splice) |

Phase 4.6 ships partial: `rm_weighted_rms_norm_seg`, `ra_add_seg`,
`fa_add_seg`, `sw_silu_mul_seg`, and the outer unnamed dispatcher
device are on placed-IRON. The 4 GEMM devices (`og_matmul_seg`,
`dg_matmul_seg`, `gg_matmul_seg`, `ug_matmul_seg`) hit a real wall in
two prior attempts (structural diff perfect, runtime garbage);
deferred pending deeper analysis (likely AIR source-of-truth).

Decode steady-state: ~7.8 tok/s on NPU2 with the current kernels (real
HF weights, `unsloth/Llama-3.2-1B-Instruct`).

### Where each part of the pipeline runs

| Stage | Runs on | Notes |
|---|---|---|
| Prefill: RMSNorm + QKV GEMM + RoPE | NPU (placed-IRON + PythoC `rope_pythoc.o` + `bf16_gemm_pythoc_*.o`) | All 7 devices on placed-IRON |
| Prefill: flash attention | NPU (placed-IRON + PythoC `attn_pythoc.o`) | All 32 cores, cascade chain |
| Prefill: O + FFN | NPU (cached MLIR + PythoC `silu_and_mul_bf16.o`) | 5 of 9 devices on placed-IRON; 4 GEMM devices cached-spliced (1024 inline `vector.contract`) |
| Decode: RMSNorm + QKV GEMV + RoPE | NPU (placed-IRON + PythoC kernels) | per-layer, per-token |
| Decode: attention compute | **CPU numpy** | LQ=1, dispatch overhead beats NPU GEMV |
| Decode: O + FFN | NPU (placed-IRON + PythoC kernels) | per-layer, per-token |
| Decode: LM head | NPU (placed-IRON + PythoC `mv_pythoc.o`) | once per token |

Decode-attention-on-NPU is not implemented in this tree (the AIR
reference also leaves it on CPU). It would be a separate builder, not
a derivative of the prefill flash_attn.

### Validation gates

- `make snapshot` — synthetic-weight per-kernel JSON regression (timing
  + correlation). Phase-to-phase diffs against the most recent snapshot
  under `tests/snapshots/`.
- `make hf-gate` — real-HF answer-level check. Asks "What is the
  capital of France?" and asserts the first 10 decode tokens
  detokenize to a string containing `paris`. Defaults to
  `unsloth/Llama-3.2-1B-Instruct` (non-gated mirror) so it runs
  without HF auth.

Every PythoC swap or builder swap must pass `make hf-gate` before
commit. Synthetic-weight snapshots have lied before (e.g. `conf=0` on
the per-lane bf16 MAC silently produces wrong dot products) — real-HF
prompts are the gold standard.

## Layout

```
llama32_1b/
├── llama32_1b_inference.py     # entry point (verbatim from AIR ref)
├── llama32_1b_prefill.py       # prefill orchestration
├── llama32_1b_decode.py        # decode orchestration (decode_attention_cpu lives here)
├── llama32_1b_reference.py     # F32 CPU reference (verify)
├── llama32_1b_weights.py       # HF weight loader
├── llama32_1b_awq_runtime.py   # (Phase 6 -- inert today)
├── Makefile                    # compile / run / verify / snapshot / hf-gate
├── kernels/                    # PythoC @aie_kernel libraries
│   ├── rms_norm.py             #  rms_norm_2048_bf16.o
│   ├── silu_and_mul.py         #  silu_and_mul_bf16.o
│   ├── rope.py                 #  rope_pythoc.o
│   ├── matvec.py               #  mv_pythoc.o (K=2048)
│   ├── matvec_k8192.py         #  mv_k8192_pythoc.o (K=8192, dg_* symbols)
│   ├── attn.py                 #  attn_pythoc.o (19 flash-attn primitives)
│   └── build.py                # compile_* helpers, one per kernel .o
├── builders/                   # placed-IRON program builders (Phase 4)
│   ├── _emit.py                # shared helpers (lock barriers, arg-type generators)
│   ├── lm_head_gemv.py         # 622 LOC
│   ├── rms_gemv_rope.py        # 883 LOC -- 6-phase decode multi-launch
│   ├── o_gemv_ffn.py           # 1377 LOC -- 8-phase decode multi-launch
│   ├── flash_attn.py           # 1105 LOC -- 32-core 4-stage cascade prefill
│   ├── rms_gemms_rope.py       # 2164 LOC -- 7-device prefill RMS+QKV-GEMM+RoPE
│   └── o_ffn.py                # 1356 LOC -- 5/9 devices placed (4 GEMM devs cached-spliced)
├── reference_mlir/             # cached AIR-emitted aie/aiex MLIR
│   ├── rms_gemv_rope.npu.air.mlir   # decode (placed-IRON has parity)
│   ├── o_gemv_ffn.npu.air.mlir      # decode (placed-IRON has parity)
│   ├── lm_head_gemv.npu.air.mlir    # decode (placed-IRON has parity)
│   ├── flash_attn.npu.air.mlir      # prefill (placed-IRON has parity)
│   ├── rms_gemms_rope.npu.air.mlir  # prefill (placed-IRON has parity)
│   └── o_ffn.npu.air.mlir           # prefill -- 5/9 devices placed; 4 GEMM devs spliced from this
├── reference_o/                # EMPTY -- all .o now PythoC-built
├── kernel_builder/             # aiecc compile + XRT cache (no aircc at runtime)
│   ├── aie_compile.py
│   ├── aie_ir_gen.py           # cached loader + placed-builder feature-flag wiring
│   ├── cache.py                # KernelCache + Profiler + XRT BO reuse
│   ├── external_kernels.py     # stage .o files (PythoC-built)
│   └── backend_presets.py
└── tests/
    ├── test_phase_snapshot.py  # synthetic verify -> JSON snapshot
    ├── test_hf_answer_gate.py  # real-HF "Paris" gate
    └── snapshots/              # per-phase baselines
```

`build_peano/` and the `*_build/` dirs at the parent level are
regenerated by `make compile` and are gitignored.

## Build & run

```bash
source ../../../../env.sh                # mlir-aie venv + XRT + paths
make compile                             # ~80s; populates build_peano/{decode,prefill}_kernel_cache/
make run                                 # default HF_MODEL_ID = unsloth/Llama-3.2-1B-Instruct
make run PROMPT="What is the capital of Spain?"
make verify N_TOKENS=10                  # F32 CPU reference diff
make profile                             # per-kernel + per-token breakdown
make snapshot                            # capture JSON; gate vs prior phase
make hf-gate                             # real-HF answer-level check (~25s)
make chat                                # interactive REPL
```

### Feature flag — A/B between cached MLIR and placed-IRON

```bash
# Use only the 3 decode builders (default for decode regressions):
PYTHOC_LLAMA_USE_PLACED_BUILDERS=lm_head_gemv,rms_gemv_rope,o_gemv_ffn make hf-gate

# Use the full set including prefill flash_attn:
PYTHOC_LLAMA_USE_PLACED_BUILDERS=flash_attn,lm_head_gemv,rms_gemv_rope,o_gemv_ffn make hf-gate

# Use the cached MLIR (no placed builders):
make hf-gate
```

The `Makefile` auto-points `PEANO_INSTALL_DIR` at the AIR-tree pip
`llvm-aie` (commit `5ed1593`); the pythoc-tree's in-tree `llvm-aie`
(commit `55604435`) crashes in `InterBlockScheduling::emitLoopRemarks`
on the RoPE-K core. Set `PYTHOC_LLAMA_PEANO` to override.

## How to add a PythoC kernel

1. Write the kernel under `kernels/<name>.py` (one or more
   `@aie_kernel` functions). The LAST function is the entry point —
   earlier ones are pulled in as helpers and exported in the same `.o`.
2. Add a build helper to `kernels/build.py` that calls
   `compile_pythoc_source(function_name=...)` with `extra_globals=`
   covering every lazy intrinsic the kernel uses (PythoC's AST visitor
   only seeds a hard-coded import list; everything else has to be
   explicitly passed in).
3. Register it in `kernel_builder/external_kernels.py::_PYTHOC_KERNELS`.
4. Sed-swap the `link_with = "<name>.o"` strings in
   `reference_mlir/*.npu.air.mlir` to point at the new `.o` (or set the
   right name when emitting from a placed-IRON builder).
5. Run `make compile && PYTHOC_LLAMA_USE_PLACED_BUILDERS=... make hf-gate`
   to confirm correctness before commit.

**Watch out**: AIE2P bf16 mac intrinsics like
`I512_I512_ACC1024_bf_mac_conf` accept a `conf` operand that selects
sub-element multiply patterns. **Use `conf=60` for per-lane bf16 MAC**
(matches `attn.py`); `conf=0` silently produces wrong dot products and
the synthetic-weights verify doesn't always catch it.

## How to add a placed-IRON builder

Each builder under `builders/<name>.py` exposes a single
`build_<name>_module(...) -> str` that returns the MLIR module text.
`kernel_builder/aie_ir_gen.py::build_<name>_ir` calls it when
`PYTHOC_LLAMA_USE_PLACED_BUILDERS=<name>` is set; otherwise the cached
MLIR under `reference_mlir/` is used. Mimic the wiring in
`build_lm_head_gemv_ir` (lines 110-119 of `aie_ir_gen.py`).

`builders/_emit.py` collects shared helpers (lock-barrier emission,
host-arg-type generation, common DMA-config patterns) that the four
existing builders share — add to it when a pattern repeats.

The structural acceptance bar for a new builder is **exact op-count
parity** against its cached `npu.air.mlir` across `aie.tile`,
`aie.lock`, `aie.buffer`, `aie.core`, `aie.flow`, `aie.memtile_dma`,
`aie.shim_dma_allocation`, `aie.cascade_flow`, `aiex.dma_configure_task_for`,
`aiex.dma_start_task`, `aiex.dma_await_task`, `aiex.dma_free_task`.
The four existing builders all match within 0 across every category.

## Hand-editing the IR

```bash
make compile
$EDITOR build_peano/decode_kernel_cache/lm_head_gemv.npu.air.mlir
rm build_peano/decode_kernel_cache/lm_head_gemv.elf
make run                                 # rebuilds lm_head_gemv.elf from your edits
```

## Prefill GEMM plan (in progress)

The two remaining prefill builders — `rms_gemms_rope` and `o_ffn` —
contain **1792 inline `vector.contract` ops** between them (768 + 1024)
implementing the BF16 GEMMs at seq_len=2048 for Q/K/V/O projections
plus the FFN gate/up/down. Today aiecc auto-lowers these directly to
AIE MAC intrinsics; no PythoC `.o` is involved.

**Plan (Option B):** write a reusable PythoC BF16 GEMM kernel and have
the two placed-IRON builders link to it as an external `.o`, the same
way `flash_attn.py` links to `attn_pythoc.o`. This:

- pushes all GEMM math into PythoC (advancing the "pure PythoC" goal),
- factors out the inline-vector.contract body into a function call per
  core (cuts the builders from O(thousands) LOC to something closer to
  the decode builders' size),
- converges the prefill design on the same external-kernel pattern as
  flash_attn and the decode builders.

Reuse candidate: `programming_examples/pythoc/bf16_gemm_multi_core.py`
already implements a working BF16 GEMM in PythoC. Step 1 is to check
whether its kernel signature + tile shapes can be adapted for the
prefill GEMM dimensions (Q/K/V at seq=2048, head_dim=64; FFN
intermediate=8192); if not, fork or extend it. Step 2 is one of the
two builders (probably `rms_gemms_rope` first since it has 75% of the
GEMM count of `o_ffn`).

## Plan & tracking

- Full project plan: `~/.claude/plans/using-flash-attention-which-is-rosy-crane.md`
- Beads epic: `PythoC-8ns` in `~/npu-dev-pythoc/PythoC`
- Phase log:
  - Phase 0-1: skeleton + cached-MLIR baseline
  - Phase 2: RMSNorm PythoC swap
  - Phase 3.1-3.3: silu_and_mul, rope, matvec, matvec_k8192 PythoC swaps
  - Phase 3.4: 19 flash-attention primitives in PythoC (`attn_pythoc.o`)
  - Phase 4.1-4.4: placed-IRON builders (lm_head_gemv, rms_gemv_rope, o_gemv_ffn, flash_attn)
  - Phase 4.5-4.6: prefill GEMM builders (in progress — see *Prefill GEMM plan* above)
  - Phase 6: AWQ uint4 path (deferred)

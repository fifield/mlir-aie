# LLAMA-3.2-1B BF16 Inference on AMD NPU2 — PythoC + placed IRON

End-to-end Llama-3.2-1B inference on AMD NPU2 (Strix Halo, aie2p), built
incrementally on top of the MLIR-AIR reference at
[`mlir-air-llama_awq_impl/programming_examples/llama32_1b_aie`][air-src].
Every kernel is now a PythoC `@aie_kernel` function and every AIR
multi-launch is now a placed-IRON (`aie/aiex`-dialect) Python builder
that runs by default (`PYTHOC_LLAMA_USE_PLACED_BUILDERS=cached` reverts
to the cached AIR-emitted MLIR substrate for A/B). One builder
(`o_ffn`) ships partial: 5 of its 9 devices are on placed-IRON; the 4
prefill GEMM devices are spliced from cached MLIR — see *Phase 4.6
status* below.

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

### Placed-IRON builders — 6 of 6 enabled by default

| Builder | Phase | Used by | Default |
|---|---|---|---|
| `builders/lm_head_gemv.py` | decode (final logits) | `llama32_1b_decode.py` | ✓ placed-IRON |
| `builders/rms_gemv_rope.py` | decode (RMSNorm + QKV GEMV + RoPE) | per-layer decode | ✓ placed-IRON |
| `builders/o_gemv_ffn.py` | decode (O + FFN) | per-layer decode | ✓ placed-IRON |
| `builders/flash_attn.py` | prefill flash attention | `llama32_1b_prefill.py` | ✓ placed-IRON |
| `builders/rms_gemms_rope.py` | prefill (RMSNorm + QKV GEMM + RoPE) | per-layer prefill | ✓ placed-IRON |
| `builders/o_ffn.py` | prefill (O + FFN with GEMMs) | per-layer prefill | ◐ placed-IRON (5 of 9 devices; 4 GEMM devices internally spliced from cached) |

`rms_gemms_rope`'s prefill V/K/Q GEMMs originally landed with a kernel
stride/loop-bound mismatch (kernel built as `M_BLOCKS=16, N_BLOCKS=8`
when the cached contract walks the L1 buffers as if `M_BLOCKS=8,
N_BLOCKS=16`). The kernel produced uncorrelated output (corr=0.007 vs
cached element-wise). Fixed by re-deriving the strides directly from
the cached contract's `arg1*64 + arg3*512` access pattern; see
`kernels/build.py::_compile_bf16_gemm_rms_gemms_rope` and
`tests/test_v_matmul_oracle.py` for the diagnostic harness.

#### Phase 4.6 status (o_ffn partial)

5 of 9 devices in `o_ffn` are on placed-IRON:
`rm_weighted_rms_norm_seg`, `ra_add_seg`, `fa_add_seg`,
`sw_silu_mul_seg`, and the outer dispatcher. The 4 GEMM devices
(`og_matmul_seg`, `dg_matmul_seg`, `gg_matmul_seg`, `ug_matmul_seg`)
are spliced from `reference_mlir/o_ffn.npu.air.mlir` by the builder
itself — transparent to call sites in `aie_ir_gen.py`. The 4 GEMM
devices hit the same class of bug as `rms_gemms_rope`'s V/K/Q GEMMs
(structural-diff-clean / runtime-garbage from a kernel-stride/L1-layout
mismatch). Now that `rms_gemms_rope` is fixed, the same diagnostic
approach (see `tests/test_v_matmul_oracle.py`) can be applied to derive
the correct kernel params for og/dg/gg/ug — they each have different
K/N shapes from v_matmul so they need their own per-device kernel
builds.

#### Performance

Real HF weights (`unsloth/Llama-3.2-1B-Instruct`), measured on NPU2:

| Config | Prefill (16 layers, seq=2048) | Decode steady-state |
|---|---|---|
| Default (6 placed builders) | ~1.91s | ~8.05 tok/s |
| All cached (`PYTHOC_LLAMA_USE_PLACED_BUILDERS=cached`) | ~1.92s | ~8.07 tok/s |

Delta within run-to-run noise.

### Where each part of the pipeline runs

| Stage | Runs on | Notes |
|---|---|---|
| Prefill: RMSNorm + QKV GEMM + RoPE | NPU (cached MLIR + PythoC `rope_pythoc.o` + `bf16_gemm_pythoc_*.o`) | Placed-IRON deferred (see *Deferred* section) |
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
│   ├── aie_ir_gen.py           # placed-builder dispatcher (cached substrate as override)
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

### Feature flag — A/B between placed-IRON (default) and cached MLIR

```bash
# Default: every builder runs placed-IRON Python emit:
make hf-gate

# Force every builder onto the cached MLIR substrate (A/B regression test):
PYTHOC_LLAMA_USE_PLACED_BUILDERS=cached make hf-gate

# Explicit allowlist (overrides the default; only these builders use placed-IRON):
PYTHOC_LLAMA_USE_PLACED_BUILDERS=lm_head_gemv,rms_gemv_rope,o_gemv_ffn make hf-gate

# "all" is identical to the default; kept for backwards-compat:
PYTHOC_LLAMA_USE_PLACED_BUILDERS=all make hf-gate
```

Default set: `lm_head_gemv`, `flash_attn`, `rms_gemv_rope`,
`o_gemv_ffn`, `rms_gemms_rope`, `o_ffn` (all 6 current builders). See
`kernel_builder/aie_ir_gen.py::_DEFAULT_PLACED_BUILDERS`.

### Runtime oracle for the prefill GEMM kernel

`tests/test_v_matmul_oracle.py` compiles two versions of `rms_gemms_rope`
(all-cached vs placed-IRON with one or more devices spliced) and diffs
the V/K/Q outputs element-by-element. Used to catch
structural-diff-clean / runtime-garbage bugs in the placed-IRON GEMM
emit. Modes: `--mode=v_only` (default), `--mode=vkq`, `--mode=full`.
Run as `cd build_peano && python3 ../tests/test_v_matmul_oracle.py
--mode=full`.

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
5. Run `make compile && make hf-gate` to confirm correctness before
   commit (placed-IRON is default; add
   `PYTHOC_LLAMA_USE_PLACED_BUILDERS=cached` to verify the cached
   path still works as an A/B reference).

**Watch out**: AIE2P bf16 mac intrinsics like
`I512_I512_ACC1024_bf_mac_conf` accept a `conf` operand that selects
sub-element multiply patterns. **Use `conf=60` for per-lane bf16 MAC**
(matches `attn.py`); `conf=0` silently produces wrong dot products and
the synthetic-weights verify doesn't always catch it.

## How to add a placed-IRON builder

Each builder under `builders/<name>.py` exposes a single
`build_<name>_module(...) -> str` that returns the MLIR module text.
`kernel_builder/aie_ir_gen.py::build_<name>_ir` calls it by default if
the name is in `_DEFAULT_PLACED_BUILDERS`; setting
`PYTHOC_LLAMA_USE_PLACED_BUILDERS=cached` forces a fallback to
`reference_mlir/<name>.npu.air.mlir`. Mimic the wiring in
`build_lm_head_gemv_ir` and add the new name to
`_DEFAULT_PLACED_BUILDERS` when the builder lands the HF gate.

`builders/_emit.py` collects shared helpers (lock-barrier emission,
host-arg-type generation, common DMA-config patterns) that the
existing builders share — add to it when a pattern repeats. The Phase
4.5/4.6 builders also share device-emit + splice helpers via direct
imports from `builders/rms_gemms_rope.py` (`_splice_device`,
`_extract_single_device`, `_emit_matmul_device`) — see
`builders/o_ffn.py` for the pattern.

The structural acceptance bar for a new builder is **exact op-count
parity** against its cached `npu.air.mlir` across `aie.tile`,
`aie.lock`, `aie.buffer`, `aie.core`, `aie.flow`, `aie.memtile_dma`,
`aie.shim_dma_allocation`, `aie.cascade_flow`, `aiex.dma_configure_task_for`,
`aiex.dma_start_task`, `aiex.dma_await_task`, `aiex.dma_free_task`.
Body deltas are expected for GEMM cores where the cached emits inline
`vector.contract` chains and the placed-IRON emit collapses them to
`func.call @bf16_gemm_kernel_bf16out`. **Structural parity is
necessary but not sufficient** — Phases 4.6d/e shipped perfect
structural matches that hung at runtime due to dispatch/DMA-streaming
differences not visible in the MLIR text. The HF gate is the gold
standard; never commit on structural diff alone.

## Hand-editing the IR

```bash
make compile
$EDITOR build_peano/decode_kernel_cache/lm_head_gemv.npu.air.mlir
rm build_peano/decode_kernel_cache/lm_head_gemv.elf
make run                                 # rebuilds lm_head_gemv.elf from your edits
```

## Deferred: 4 prefill GEMM devices in o_ffn

Phase 4.6's `og_matmul_seg`, `dg_matmul_seg`, `gg_matmul_seg`, and
`ug_matmul_seg` are spliced from cached MLIR by `builders/o_ffn.py`.
Three attempts (Phases 4.6d twice and 4.6e once) achieved **exact
structural op-count parity** vs cached — same tiles, locks (with same
init values), buffers, flows, memtile_dmas, shim_dma_allocations, and
runtime_sequence configures — but the runtime produced garbage tokens
(e.g. `, 0, 0, 10,` instead of `Paris`).

**Root cause was the same class of bug as `rms_gemms_rope::v_matmul_seg`**:
the GEMM kernel was being built with stride/loop-bound params that
didn't match how the cached contract walks the L1 buffer. For
`v_matmul_seg`, the kernel was configured as `M_BLOCKS=16/N_BLOCKS=8`
but the cached contract actually walks the buffers as `M_BLOCKS=8/
N_BLOCKS=16` with K-stride 512 (not 256) and N-stride 256 (not 64).
See `kernels/build.py::_compile_bf16_gemm_rms_gemms_rope` for the
verified derivation against `reference_mlir/rms_gemms_rope.npu.air.mlir`.

`og/dg/gg/ug` each have different K/N tile shapes from v_matmul, so
they each need a separate kernel build with strides derived from
their respective cached contract access patterns. The diagnostic
harness at `tests/test_v_matmul_oracle.py` can be adapted per-device
(splice ONE device's placed-IRON emit into otherwise-cached
`o_ffn.npu.air.mlir`, diff the device's output buffer vs all-cached).

**Full multi-session brief** lives in beads at `PythoC-8ns.13` (see
`bd show PythoC-8ns.13` in `~/npu-dev-pythoc/PythoC`).

The 4 deferred devices being on the cached MLIR substrate is
end-to-end-equivalent to the pre-Phase-4.6 state for the
`o_ffn`-portion of prefill. No correctness regression; only the
"every core's MLIR is emitted from placed-IRON Python" milestone is
unmet for these 4 devices.

## Plan & tracking

- Full project plan: `~/.claude/plans/using-flash-attention-which-is-rosy-crane.md`
- Beads epic: `PythoC-8ns` in `~/npu-dev-pythoc/PythoC`
- Phase log:
  - Phase 0-1: skeleton + cached-MLIR baseline
  - Phase 2: RMSNorm PythoC swap
  - Phase 3.1-3.3: silu_and_mul, rope, matvec, matvec_k8192 PythoC swaps
  - Phase 3.4: 19 flash-attention primitives in PythoC (`attn_pythoc.o`)
  - Phase 4.1: `lm_head_gemv` placed-IRON
  - Phase 4.2: `flash_attn` placed-IRON
  - Phase 4.3: `rms_gemv_rope` placed-IRON
  - Phase 4.4: `o_gemv_ffn` placed-IRON
  - Phase 4.5 (a→e): `rms_gemms_rope` placed-IRON, 7 of 7 devices
  - Phase 4.6 (a→c, f): `o_ffn` placed-IRON, 5 of 9 devices; 4 GEMM devices deferred (see *Deferred* above)
  - Phase 6: AWQ uint4 path (deferred)
- Defaults: placed-IRON for every shipped builder. Override with
  `PYTHOC_LLAMA_USE_PLACED_BUILDERS=cached` to A/B against the cached
  MLIR substrate.

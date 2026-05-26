# LLAMA-3.2-1B BF16 Inference on AMD NPU2 — PythoC + placed IRON

End-to-end Llama-3.2-1B inference on AMD NPU2 (Strix Halo, aie2p), built
incrementally on top of the MLIR-AIR reference at
[`mlir-air-llama_awq_impl/programming_examples/llama32_1b_aie`][air-src].
Every kernel is now a PythoC `@aie_kernel` function and every AIR
multi-launch is now a placed-IRON (`aie/aiex`-dialect) Python builder
that runs by default (`PYTHOC_LLAMA_USE_PLACED_BUILDERS=cached` reverts
to the cached AIR-emitted MLIR substrate for A/B). All 6 builders ship
with all of their devices on placed-IRON — including `o_ffn`'s 4
prefill GEMM devices (og/gg/ug/dg) landed in Phase 4.6d/e after the
v_matmul kernel-stride bug fix in Phase 4.5 unblocked the
diagnostic approach.

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
| AWQ uint4 matvec (fused decode, runtime m/k) | o_gemv_ffn_awq | `kernels/awq_mv.py` + `kernels/awq_mv_k8192.py` | `awq_mv_pythoc.o`, `awq_mv_k8192_pythoc.o` ✓ |
| AWQ uint4 GEMV (standalone, dim-specialized) | awq_matvec | `kernels/awq_gemv_k{2048_m32,8192_m8}_g128_vecdeq.py` | `awq_gemv_k{2048_m32,8192_m8}_g128_vecdeq_pythoc.o` ✓ |

`reference_o/` is empty — no `.cc`-built `.o` left in the project.

### Placed-IRON builders — 10 of 10 enabled by default

| Builder | Phase | Used by | Default |
|---|---|---|---|
| `builders/lm_head_gemv.py` | decode (final logits) | `llama32_1b_decode.py` | ✓ placed-IRON |
| `builders/rms_gemv_rope.py` | decode (RMSNorm + QKV GEMV + RoPE) | per-layer decode | ✓ placed-IRON |
| `builders/o_gemv_ffn.py` | decode (O + FFN) | per-layer decode | ✓ placed-IRON |
| `builders/flash_attn.py` | prefill flash attention | `llama32_1b_prefill.py` | ✓ placed-IRON |
| `builders/rms_gemms_rope.py` | prefill (RMSNorm + QKV GEMM + RoPE) | per-layer prefill | ✓ placed-IRON |
| `builders/o_ffn.py` | prefill (O + FFN with GEMMs) | per-layer prefill | ✓ placed-IRON (all 9 devices, incl. og/gg/ug/dg GEMMs landed Phase 4.6d/e) |
| `builders/o_gemv_ffn_awq.py` | decode (O + FFN, packed-AWQ) | `o_gemv_ffn_awq_npu` (when `--quant awq`) | ✓ placed-IRON (Phase 6 Stage 3) |
| `builders/awq_matvec.py` | standalone AWQ GEMV | `awq_gemv_npu`, `awq_gemv_npu_tiled` | ✓ placed-IRON (Phase 6 Stage 3) |
| `builders/lm_head_gemv_awq.py` | decode (final logits, packed-AWQ) | LM head when `--quant awq` | ✓ placed-IRON (Phase 6 follow-up) |
| `builders/rms_gemv_rope_awq.py` | decode (RMSNorm + Q/K/V GEMV + RoPE, packed-AWQ) | per-layer decode when `--quant awq` | ✓ placed-IRON (Phase 6 follow-up) |

`rms_gemms_rope`'s prefill V/K/Q GEMMs originally landed with a kernel
stride/loop-bound mismatch (kernel built as `M_BLOCKS=16, N_BLOCKS=8`
when the cached contract walks the L1 buffers as if `M_BLOCKS=8,
N_BLOCKS=16`). The kernel produced uncorrelated output (corr=0.007 vs
cached element-wise). Fixed by re-deriving the strides directly from
the cached contract's `arg1*64 + arg3*512` access pattern; see
`kernels/build.py::_compile_bf16_gemm_rms_gemms_rope` and
`tests/test_v_matmul_oracle.py` for the diagnostic harness.

#### Phase 4.6 status — complete

All 9 devices in `o_ffn` are now placed-IRON:
`rm_weighted_rms_norm_seg`, `ra_add_seg`, `fa_add_seg`,
`sw_silu_mul_seg`, the outer dispatcher, **plus the 4 GEMM devices
(`og_matmul_seg`, `gg_matmul_seg`, `ug_matmul_seg`, `dg_matmul_seg`)**
landed in Phase 4.6d/e using the kernel-stride diagnostic approach
established by Phase 4.5. Per-device kernel build params + per-device
splice oracle (`tests/test_v_matmul_oracle.py --mode={og,gg,ug,dg}`)
verified each device bit-identical vs the cached AIR contract chain.

The 4 GEMMs share two kernel object files (gg/ug reuse the v_matmul
M=8/N=16 kernel since their per-core body is byte-identical; og/dg
share a M=8/N=8 kernel that differs only in the host-side outer loop
count). See `kernels/build.py` for the build helpers and
`builders/o_ffn.py::_emit_{og,gg,ug,dg}_matmul_seg` for the per-device
emits.

#### Performance

Real HF weights (`unsloth/Llama-3.2-1B-Instruct`), measured on NPU2:

| Config | Prefill (16 layers, seq=2048) | Decode steady-state |
|---|---|---|
| BF16 default (6 placed builders, all 9 o_ffn devices placed) | ~1.84s | ~8.19 tok/s (122 ms/token) |
| BF16 all cached (`PYTHOC_LLAMA_USE_PLACED_BUILDERS=cached`) | ~1.92s | ~8.08 tok/s |
| AWQ chunk-unrolled + fused-MAC dequant (current) | ~1.89s | ~8.22 tok/s (122 ms/token) |
| AWQ pre-optimization (chunk-looped, sub+mul dequant) | ~1.89s | ~6.43 tok/s (155 ms/token) |
| AWQ scalar per-nibble (kept for reference) | ~1.85s | ~0.06 tok/s (16952 ms/token) |

AWQ decode is now **at BF16 parity** (~122 ms/token, 8.22 tok/s).  Two
follow-on optimizations on top of the Fix2Float vectorization landed
the gap to zero:

* **Chunk-loop unroll** — `chunks_per_group=2` for GROUP_SIZE=128 is a
  compile-time constant, so the inner 2-iter chunk loop is fully
  inlined into the per-group body.  The pipeliner can't pipeline a
  2-iter loop (prologue+epilogue dominate), so inlining halves the
  per-group bundle count by removing the loop branch / phi / re-init
  overhead.  The per-group basic block goes from ~112 to ~60 bundles.
  Group-loop pipelining via `prepare_for_pipelining()` is *not*
  applied — the 4-acc unrolling needed to break the MAC accumulator
  recurrence triggers register-spill miscompiles on AIE2P, so we keep
  2 accumulators and let the existing scheduling work.
* **MAC+MSC math fusion** — replace `(w_bf - zero) * scale` (vsub +
  vmul + an accfloat<->bf16 round-trip per chunk-half) with
  `w_scaled = w_bf * scale; acc = mac(x, w_scaled); acc = msc(x, zs)`
  where `zs = zero * scale` is precomputed per group (scalar bf16,
  broadcast to 32-lane).  The mac+msc pair keeps everything in
  accfloat without intermediate bf16 conversions, saving the
  conversion latency.

The uint4→bf16 inner loop still uses the AIE-API Fix2Float
magic-number reinterpret trick (aie_api/detail/aie2p/elementary.hpp:51-58):
unpack u4 nibbles to u8, zero-extend to acc32 via UPS, integer-add
the magic constant `0x4b010000` per lane, bitcast acc32 → accfloat,
then subtract the magic in bf16 space via the
`bf_msc_conf(magic_bf, 1.0, acc, conf=60)` hardware multiply-subtract
(folds the float-subtract into a MAC unit).  The `<32 x f32>`
fadd/fsub vector ops don't legalize on AIE2P GISel; the MSC trick
avoids that.  AWQ correctness verified by `make hf-gate QUANT=awq`
(decodes to "The capital of France is Paris.").

#### Phase 6 status — complete

AWQ uint4 decode path lands end-to-end:
- PythoC kernels (`kernels/awq_mv.py`, `awq_mv_k8192.py`, dim-specialized
  `awq_gemv_k{K}_m{M}_g{G}_vecdeq.py`) replace the C++ `awq_mv.cc` /
  `awq_gemv.cc` from the awq_impl branch.
- Placed-IRON builders (`builders/o_gemv_ffn_awq.py`,
  `builders/awq_matvec.py`) replace the AIR-tree `awq_matvec.py` /
  `awq_gemv_builder.py` stitchers.
- All scaffolding (.cc + AIR builders + multi-launch stitcher) deleted
  in Stage 4. Cached MLIR retained as A/B fallback via
  `PYTHOC_LLAMA_USE_PLACED_BUILDERS=cached`.
- Default-on: `make run-awq` automatically routes through NPU AWQ
  decode (was opt-in via `AWQ_DECODE_EXPERIMENTAL=1` pre-Phase 6).
- HF-gate AWQ + BF16 no-regression + A/B cached fallback all green on
  real `unsloth/Llama-3.2-1B-Instruct` AWQ weights.

### Where each part of the pipeline runs

| Stage | Runs on | Notes |
|---|---|---|
| Prefill: RMSNorm + QKV GEMM + RoPE | NPU (placed-IRON + PythoC `rope_pythoc.o` + `bf16_gemm_pythoc_*.o`) | All 7 devices placed-IRON |
| Prefill: flash attention | NPU (placed-IRON + PythoC `attn_pythoc.o`) | All 32 cores, cascade chain |
| Prefill: O + FFN | NPU (placed-IRON + PythoC `silu_and_mul_bf16.o` + `bf16_gemm_pythoc_*.o`) | All 9 devices placed-IRON (incl. og/gg/ug/dg GEMMs) |
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
│   └── o_ffn.py                # ~3700 LOC -- all 9 devices placed (incl. og/gg/ug/dg GEMMs)
├── reference_mlir/             # cached AIR-emitted aie/aiex MLIR
│   ├── rms_gemv_rope.npu.air.mlir   # decode (placed-IRON has parity)
│   ├── o_gemv_ffn.npu.air.mlir      # decode (placed-IRON has parity)
│   ├── lm_head_gemv.npu.air.mlir    # decode (placed-IRON has parity)
│   ├── flash_attn.npu.air.mlir      # prefill (placed-IRON has parity)
│   ├── rms_gemms_rope.npu.air.mlir  # prefill (placed-IRON has parity)
│   └── o_ffn.npu.air.mlir           # prefill -- ground truth oracle (placed-IRON has parity)
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

## Prefill GEMM diagnostic methodology (Phase 4.5/4.6 lessons)

`rms_gemms_rope`'s V/K/Q GEMMs (Phase 4.5) and `o_ffn`'s 4 GEMM
devices (Phase 4.6d/e, `og/gg/ug/dg`) all hit the same class of bug
on first attempts: **structural op-count diff perfect vs cached, but
runtime produced garbage tokens**. Op-count diff cannot detect that
the bf16 GEMM kernel's compile-time stride/loop-bound constants are
inconsistent with how the cached AIR-emitted contract chain walks the
L1 buffers.

The fix in each case is to **derive the kernel build params directly
from the cached contract's `vector.transfer_read` offset expressions**
(typically `arg_m*64 + arg_k*512` for the LHS, `arg_n*256 + arg_k*64`
for the RHS, etc.) and to verify bit-exact output via a runtime
oracle that splices ONE placed device into otherwise-cached MLIR and
diffs the output buffer element-by-element. The oracle lives at
`tests/test_v_matmul_oracle.py`; invoke with `--mode={v_only,vkq,
full,og,gg,ug,dg}`.

Per-device kernel objects (in `kernels/build.py`):

| Device(s) | Kernel object | M_BLOCKS / N_BLOCKS / K_MICRO |
|---|---|---|
| `v/k/q_matmul_seg`, `gg/ug_matmul_seg` | `bf16_gemm_pythoc_M8_N16_K4_AT_bf16out_s64_512_64_256_64_512.o` | 8 / 16 / 4 |
| `og/dg_matmul_seg` | `bf16_gemm_pythoc_M8_N8_K4_AT_bf16out_s64_512_64_256_64_512.o` | 8 / 8 / 4 |

Both kernels share the same per-call strides (`A_M=64, A_K=512,
B_K=64, B_N=256, C_M=64, C_N=512`); only the loop bounds differ.
Devices that share a kernel differ at the dispatch / host-arg /
shim-channel / runtime-sequence level (e.g. gg vs v_matmul has 4×
more N-dispatches per core because gg's output is N=8192 vs v's
N=512). See `builders/o_ffn.py::_emit_{og,gg,ug,dg}_matmul_seg` and
`builders/rms_gemms_rope.py::_emit_matmul_device` for the
per-device emit shapes.

Full multi-session brief and resolution notes live in beads at
`PythoC-8ns.13` (see `bd show PythoC-8ns.13` in
`~/npu-dev-pythoc/PythoC`).

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
  - Phase 4.5 (a→e): `rms_gemms_rope` placed-IRON, 7 of 7 devices (V/K/Q GEMM stride fix landed post-4.5e)
  - Phase 4.6 (a→f): `o_ffn` placed-IRON, all 9 of 9 devices (og/gg/ug/dg GEMMs landed Phase 4.6d/e via per-device kernel-stride derivation)
  - Phase 6 (Stages 0-4): AWQ uint4 path -- PythoC kernels +
    placed-IRON builders, default-on for `--quant awq`. HF-gate green.
- Defaults: placed-IRON for every shipped builder. Override with
  `PYTHOC_LLAMA_USE_PLACED_BUILDERS=cached` to A/B against the cached
  MLIR substrate.

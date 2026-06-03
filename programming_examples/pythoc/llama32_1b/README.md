# LLAMA-3.2-1B Inference on AMD NPU2 — PythoC + placed IRON

End-to-end Llama-3.2-1B inference on AMD NPU2 (Strix Halo, aie2p).
Every kernel is a PythoC `@aie_kernel` function (compiled to `.o`); every
multi-launch is a placed-IRON Python builder that emits `aie/aiex` MLIR
directly. No `aircc` at runtime; aiecc compiles each kernel against the
PythoC-built `.o` files. BF16 and AWQ uint4 decode both supported.

The AIR-tree reference at
[`mlir-air-llama_awq_impl/programming_examples/llama32_1b_aie`][air-src]
is the parity oracle. The cached MLIR under `reference_mlir/` is kept as
an A/B fallback (set `PYTHOC_LLAMA_USE_PLACED_BUILDERS=cached` to force
every builder onto it).

[air-src]: ../../../../mlir-air-llama_awq_impl/programming_examples/llama32_1b_aie

## Current performance

Real HF weights (`unsloth/Llama-3.2-1B-Instruct`) on NPU2.

| Path | tok/s | ms/token |
|---|---|---|
| BF16 (packed decode, default) | **11.4** | 88 |
| BF16 (unpacked baseline) | 10.75 | 93 |
| AWQ uint4 | **9.60** | 104 |

BF16 decode device-packing (`d1d3d4` / `rgr2_ddr`) is now default-on and
gives +6.7% over the unpacked baseline on an A/B with bit-identical tokens
(see "Decode device-packing" below and `DEVICE_PACKING_ANALYSIS.md`
§15). The BF16 rows are a single answer-length run for the standard
"capital of France" prompt; the AWQ row predates the packing work and is
still the prior 100-token / 5-run median. AWQ decode is not yet packed.

Correctness gated by `make hf-gate` (asserts decode contains "paris" for
"What is the capital of France?") on real HF weights, for both the packed
default and the unpacked baseline.

## What runs where

| Stage | Runs on | Notes |
|---|---|---|
| Prefill: RMSNorm + QKV GEMM + RoPE | NPU placed-IRON + PythoC `bf16_gemm_pythoc_*.o` | 7-device multi-launch |
| Prefill: flash attention | NPU placed-IRON + `attn_pythoc.o` | 32 cores, 4-stage cascade |
| Prefill: O + FFN | NPU placed-IRON + `silu_and_mul_bf16.o` + `bf16_gemm_pythoc_*.o` | 9 devices, includes og/gg/ug/dg GEMMs |
| Decode: RMSNorm + QKV GEMV + RoPE | NPU placed-IRON + `rms_norm_2048_bf16.o` + `mv_pythoc.o` + `rope_pythoc.o` | per-layer, per-token |
| Decode: attention | **CPU numpy** | LQ=1, dispatch overhead beats NPU GEMV; AIR ref also leaves it on CPU |
| Decode: O + FFN | NPU placed-IRON + `mv_pythoc.o` + `mv_k8192_pythoc.o` + `silu_and_mul_bf16.o` | per-layer, per-token |
| Decode: LM head | NPU placed-IRON + `mv_pythoc.o` (or `awq_mv_pythoc.o`) | 8 partitions × N_OUTER=16 outer iters per token |

AWQ uint4 decode swaps the matvec kernels (`awq_mv_pythoc.o`,
`awq_mv_k8192_pythoc.o`) and routes through `rms_gemv_rope_awq` /
`o_gemv_ffn_awq` / `lm_head_gemv_awq`. Default-on for `--quant awq`.

## Kernels (PythoC)

| Kernel | Where used | Source | Built `.o` |
|---|---|---|---|
| RMSNorm K=2048 (decode) | rms_gemv_rope* | `kernels/rms_norm.py` | `rms_norm_2048_bf16.o` |
| silu_and_mul (SwiGLU) | o_gemv_ffn*, o_ffn | `kernels/silu_and_mul.py` | `silu_and_mul_bf16.o` |
| rope (half-split) | rms_gemv_rope*, rms_gemms_rope | `kernels/rope.py` | `rope_pythoc.o` |
| matvec K=2048 BF16 | rms_gemv_rope, o_gemv_ffn, lm_head_gemv | `kernels/matvec.py` | `mv_pythoc.o` |
| matvec K=8192 BF16 | o_gemv_ffn (FFN down) | `kernels/matvec_k8192.py` | `mv_k8192_pythoc.o` |
| Flash-attention primitives (19) | flash_attn | `kernels/attn.py` | `attn_pythoc.o` |
| BF16 GEMM (prefill matmuls) | rms_gemms_rope, o_ffn | `kernels/bf16_gemm.py` | `bf16_gemm_pythoc_*.o` |
| AWQ uint4 matvec (fused decode) | rms_gemv_rope_awq, o_gemv_ffn_awq, lm_head_gemv_awq | `kernels/awq_mv.py`, `awq_mv_k8192.py` | `awq_mv_pythoc.o`, `awq_mv_k8192_pythoc.o` |
| AWQ uint4 GEMV (dim-specialized) | awq_matvec | `kernels/awq_gemv_k{2048,8192}_*_vecdeq.py` | `awq_gemv_*_pythoc.o` |

## Placed-IRON builders

All 10 builders default to placed-IRON emit:

| Builder | Stage |
|---|---|
| `builders/lm_head_gemv.py` | decode LM head |
| `builders/rms_gemv_rope.py` | decode QKV GEMV + RoPE |
| `builders/o_gemv_ffn.py` | decode O + FFN |
| `builders/flash_attn.py` | prefill flash attention |
| `builders/rms_gemms_rope.py` | prefill QKV GEMM + RoPE |
| `builders/o_ffn.py` | prefill O + FFN |
| `builders/lm_head_gemv_awq.py` | decode LM head (AWQ) |
| `builders/rms_gemv_rope_awq.py` | decode QKV GEMV + RoPE (AWQ) |
| `builders/o_gemv_ffn_awq.py` | decode O + FFN (AWQ) |
| `builders/awq_matvec.py` | standalone AWQ GEMV |

## Optimization status

The matvec sub-devices have been progressively tuned through trace-driven
optimization. Full per-tile state (W L1/L2 sizes, X L1 size, ping-pong
state for each role) lives in [`PINGPONG_STATUS.md`](PINGPONG_STATUS.md).

**Headline findings from the trace work:**

1. **Bigger tiles beat ping-pong on every kernel tested.** Both attack
   per-K-iter overhead; bigger tiles do it more efficiently — fewer
   BD/lock cycles, lower L1 footprint, less DMA channel contention. All
   matvec sub-devices now run with `K_TILE = M_TILE` (single-iter K-loop).

2. **The dequant chain is the limiter for AWQ.** Trace shows `vec_util`
   stuck at 15-19% even after the DMA path is fully saturated;
   `lock_stall` and DMA starvation are already low. The bigger-tile
   wins on AWQ come from reducing X-load count, not from helping
   compute.

3. **LM head matters more than expected.** 8 partitions × 16 outer iters
   = 128 dispatch iters per token. Single-knob `K_TILE 4 → 8` change
   moved BF16 by +0.39 tok/s alone.

4. **Device packing cuts per-token dispatch ~55%.** Packing decode phases
   into fewer `aie.device` blocks drops segment dispatches 232 → 104 per
   token (`o_gemv_ffn` 8 → 4, `rms_gemv_rope` 6 → 2), bit-exact and gated.
   **Now default-on** (`d1d3d4` / `rgr2_ddr`); see the "Decode
   device-packing" section above and `DEVICE_PACKING_ANALYSIS.md` §14/§15.

Current default state per matvec role:

| Role | BF16 default | AWQ default |
|---|---|---|
| K=2048 matvecs (V/Q/K, og/gg/ug, LM head) | `K_TILE = 8` | `K_TILE = 8` |
| K=8192 matvec (dg FFN-down) | `K_TILE_K8192 = 2` + L2 W ping-pong | `K_TILE_K8192 = 2` |

Several ping-pong infrastructures are plumbed in the builders but off
by default — they were tried and found neutral or net-negative compared
to the bigger-tile approach. See `PINGPONG_STATUS.md` for the full
matrix.

### Trace-driven tuning

```bash
make trace KERNEL=rms_gemv_rope SUBDEVICE=v_matvec_bf16_0 COL=0 ROW=2
# packed default: trace a fused device (d1/d3/d4_*_pack), not the old per-matvec names
make trace KERNEL=o_gemv_ffn_awq SUBDEVICE=d4_dg_a2_pack COL=0 ROW=2 \
    QUANT=awq AWQ_WEIGHTS=/path/to/awq_repacked
# trace_summary.py aggregates a SWEEP dir (per-target subdirs from trace_sweep.py):
python3 trace_summary.py build_peano/trace_sweep/<sweep>  # summary CSV/MD
```

The trace target instruments exactly one named sub-device with AIE
hardware trace events (vec_util, lock_stall, DMA channel activity,
DMA stream starvation, memory_stall). Other kernels reuse cached
ELFs. Output:
`build_peano/{decode,prefill}_kernel_cache{,_awq}/trace/{trace.json,raw_trace.txt,meta.json}`.

`trace_summary.py` aggregates a sweep directory of per-target trace
JSONs and prints cycles-in-state per event, plus derived metrics:
`vec_util %`, `lock_stall %`, `port_running_{0,1,2} %`, `starv0/1 %`,
`dma_in0/1_eff %`.

## Build & run

```bash
source ../../../../env.sh                # mlir-aie venv + XRT + paths
make compile                             # ~30s; populates build_peano/{decode,prefill}_kernel_cache/
make run                                 # default HF_MODEL_ID = unsloth/Llama-3.2-1B-Instruct
make profile                             # per-token timing breakdown
make verify N_TOKENS=10                  # F32 CPU reference diff
make snapshot                            # JSON regression snapshot
make hf-gate                             # real-HF answer-level gate (~25s)
make chat                                # interactive REPL

# AWQ path (requires repacked AWQ weights directory)
make run-awq AWQ_WEIGHTS=/path/to/awq_repacked
make verify-awq AWQ_WEIGHTS=/path/to/awq_repacked
```

### Override the builder choice

```bash
# Default: every builder runs its placed-IRON emit
make hf-gate

# Force every builder onto the cached MLIR substrate (A/B regression):
PYTHOC_LLAMA_USE_PLACED_BUILDERS=cached make hf-gate

# Explicit allowlist:
PYTHOC_LLAMA_USE_PLACED_BUILDERS=lm_head_gemv,o_gemv_ffn make hf-gate
```

### Decode device-packing (default-on)

The two large per-token decode kernels pack multiple phases into fewer
`aie.device` blocks to cut per-token segment-dispatch reconfiguration
(232 → 104 segs/token, ~55%). These are **on by default** at the proven
validated modes (see `DEVICE_PACKING_ANALYSIS.md` §14/§15 — bit-exact
tokens, passes `hf-gate`):

| Kernel | Env var | Default | Effect |
|---|---|---|---|
| `o_gemv_ffn` (BF16) | `PYTHOC_LLAMA_O_GEMV_FFN_PACK_MODE` | `d1d3d4` | 8 → 4 devices/layer |
| `rms_gemv_rope` (BF16) | `PYTHOC_LLAMA_RMS_GEMV_ROPE_PACK_MODE` | `rgr2_ddr` | 6 → 2 devices/layer |
| `o_gemv_ffn_awq` (AWQ) | `PYTHOC_LLAMA_O_GEMV_FFN_AWQ_PACK_MODE` | `d1d3d4_rms` | 8 → 3 devices/layer (RMS folded) |
| `rms_gemv_rope_awq` (AWQ) | `PYTHOC_LLAMA_RMS_GEMV_ROPE_AWQ_PACK_MODE` | `rgr2_ddr` | 6 → 2 devices/layer |

The AWQ packs mirror the BF16 ones with packed-uint4 weights; the AWQ
A/B (`DEVICE_PACKING_ANALYSIS.md` §16) measured **9.90 → 10.75 tok/s
(+8.6%)** on real HF weights, bit-correct through `make hf-gate
QUANT=awq`.

Override (or revert to the unpacked baseline) per kernel:

```bash
# Revert both BF16 decode kernels to their unpacked single-device baseline:
PYTHOC_LLAMA_O_GEMV_FFN_PACK_MODE=none \
PYTHOC_LLAMA_RMS_GEMV_ROPE_PACK_MODE=none make profile

# Same for the AWQ decode kernels:
PYTHOC_LLAMA_O_GEMV_FFN_AWQ_PACK_MODE=none \
PYTHOC_LLAMA_RMS_GEMV_ROPE_AWQ_PACK_MODE=none make profile-awq AWQ_WEIGHTS=/path
```

For the BF16 kernels the resolved pack mode is recorded in the decode
kernel-cache manifest, so toggling those flags auto-rebuilds the affected
ELFs on the next `make run`/`make profile`/`make hf-gate` — no manual
`rm -f` needed. The AWQ decode kernels are compiled lazily at runtime
(`_preload_decode_weights`), so toggling the AWQ flags rebuilds them on the
next run directly. Resolution lives in
`kernel_builder/aie_ir_gen.py` (`decode_pack_modes` / `_resolve_pack_mode`).

### Hand-editing the cached IR

```bash
make compile
$EDITOR build_peano/decode_kernel_cache/lm_head_gemv.npu.air.mlir
rm build_peano/decode_kernel_cache/lm_head_gemv.elf
make run                                 # re-links lm_head_gemv.elf from the edited IR
```

## Validation

- `make snapshot` — synthetic-weight per-kernel JSON regression (timing
  + correlation), diffs against the most recent snapshot under
  `tests/snapshots/`.
- `make hf-gate` — real-HF "What is the capital of France?" gate.
  Asserts the first 10 decode tokens contain `paris`. Defaults to
  `unsloth/Llama-3.2-1B-Instruct` (non-gated mirror).
- `tests/test_v_matmul_oracle.py` — runtime oracle for prefill GEMMs.
  Splices one placed-IRON device into otherwise-cached MLIR and diffs
  the output element-by-element. Use modes `v_only`, `vkq`, `full`,
  `og`, `gg`, `ug`, `dg` to validate per-device GEMM emits.

Every PythoC swap or builder change must pass `make hf-gate` before
commit. Synthetic-weight snapshots have lied (e.g. AIE2P bf16 MAC
intrinsic `conf=0` silently produces wrong dot products); the real-HF
gate is the gold standard.

## Layout

```
llama32_1b/
├── llama32_1b_inference.py     # entry point (BF16 + AWQ)
├── llama32_1b_prefill.py       # prefill orchestration
├── llama32_1b_decode.py        # decode orchestration (decode_attention_cpu lives here)
├── llama32_1b_reference.py     # F32 CPU reference (verify)
├── llama32_1b_weights.py       # HF weight loader
├── llama32_1b_awq_runtime.py   # AWQ-specific runtime glue
├── Makefile                    # compile / run / verify / snapshot / hf-gate / trace
├── PINGPONG_STATUS.md          # per-tile ping-pong + tile-size state matrix
├── kernels/                    # PythoC @aie_kernel libraries
│   ├── rms_norm.py             #  rms_norm_2048_bf16.o
│   ├── silu_and_mul.py         #  silu_and_mul_bf16.o
│   ├── rope.py                 #  rope_pythoc.o
│   ├── matvec.py               #  mv_pythoc.o (K=2048)
│   ├── matvec_k8192.py         #  mv_k8192_pythoc.o (K=8192, dg_ symbols)
│   ├── attn.py                 #  attn_pythoc.o (19 flash-attn primitives)
│   ├── bf16_gemm.py            #  bf16_gemm_pythoc_*.o (prefill GEMMs)
│   ├── awq_mv.py               #  awq_mv_pythoc.o (AWQ K=2048)
│   ├── awq_mv_k8192.py         #  awq_mv_k8192_pythoc.o (AWQ K=8192)
│   ├── awq_gemv_k{2048,8192}_*_vecdeq.py   # standalone AWQ GEMV variants
│   └── build.py                # compile_* helpers, one per .o
├── builders/                   # placed-IRON program builders
│   ├── _emit.py                # shared helpers
│   ├── lm_head_gemv.py
│   ├── lm_head_gemv_awq.py
│   ├── rms_gemv_rope.py
│   ├── rms_gemv_rope_awq.py
│   ├── o_gemv_ffn.py
│   ├── o_gemv_ffn_awq.py
│   ├── flash_attn.py
│   ├── rms_gemms_rope.py
│   ├── o_ffn.py
│   └── awq_matvec.py
├── reference_mlir/             # cached AIR-emitted aie/aiex MLIR (A/B fallback)
├── kernel_builder/             # aiecc compile + XRT cache (no aircc at runtime)
│   ├── aie_compile.py
│   ├── aie_ir_gen.py           # placed-builder dispatcher
│   ├── cache.py                # KernelCache + Profiler + XRT BO reuse
│   ├── external_kernels.py     # stages PythoC .o files
│   ├── aie_trace_capture.py    # AIE hardware-trace host plumbing
│   ├── aie_trace_instrument.py # IR-level trace op injection
│   └── backend_presets.py
├── trace_summary.py            # aggregate per-target trace.json → metrics
├── trace_sweep.py              # repeatable trace sweep across targets
├── trace_sweep.targets         # default per-tile trace targets
└── tests/
    ├── test_phase_snapshot.py  # synthetic verify → JSON snapshot
    ├── test_hf_answer_gate.py  # real-HF "Paris" gate
    ├── test_v_matmul_oracle.py # per-device GEMM splice oracle
    └── snapshots/              # baselines
```

`build_peano/` and the `*_build/` dirs are regenerated by `make compile`
and are gitignored.

## How to add a PythoC kernel

1. Write the kernel under `kernels/<name>.py` (one or more
   `@aie_kernel` functions). The LAST function is the entry point;
   earlier ones are pulled in as helpers and exported in the same `.o`.
2. Add a build helper to `kernels/build.py` that calls
   `compile_pythoc_source(function_name=...)` with `extra_globals=`
   covering every lazy intrinsic the kernel uses (PythoC's AST visitor
   only seeds a hard-coded import list; everything else has to be
   passed in explicitly).
3. Register it in `kernel_builder/external_kernels.py::_PYTHOC_KERNELS`.
4. Update the `link_with = "<name>.o"` references — either in
   `reference_mlir/*.npu.air.mlir` or at the placed-IRON builder's
   `external_func(..., link_with=...)` call site.
5. `make compile && make hf-gate` before commit.

**Watch out**: AIE2P bf16 mac intrinsics like
`I512_I512_ACC1024_bf_mac_conf` take a `conf` operand that selects
sub-element multiply patterns. **Use `conf=60` for per-lane bf16 MAC**.
`conf=0` silently produces wrong dot products and synthetic verify
often misses it.

## How to add a placed-IRON builder

Each builder under `builders/<name>.py` exposes
`build_<name>_module(...) -> str` returning MLIR module text.
`kernel_builder/aie_ir_gen.py::build_<name>_ir` dispatches to it when
the name is in `_DEFAULT_PLACED_BUILDERS`. Setting
`PYTHOC_LLAMA_USE_PLACED_BUILDERS=cached` forces a fallback to
`reference_mlir/<name>.npu.air.mlir`.

`builders/_emit.py` collects shared helpers (lock-barrier emission,
host-arg-type generation, common DMA-config patterns). The
GEMM-emitting builders share device-emit + splice helpers via direct
imports between `builders/rms_gemms_rope.py` and `builders/o_ffn.py`.

**Validation bar:** structural op-count parity against the cached
`npu.air.mlir` is necessary but **not sufficient**. Builders with
perfect structural matches have shipped with runtime-garbage bugs
caused by dispatch/DMA-streaming differences not visible in the MLIR
text. The HF gate is the gold standard; for GEMM builders, additionally
run `tests/test_v_matmul_oracle.py --mode={target}` for bit-exact diff
vs the cached substrate.

## Prefill GEMM kernel constants

Two per-device kernel objects cover all 5 prefill GEMMs (`v/k/q_matmul`
in rms_gemms_rope, `og/gg/ug/dg_matmul` in o_ffn). They share strides
(`A_M=64, A_K=512, B_K=64, B_N=256, C_M=64, C_N=512`); only the
loop bounds differ.

| Device(s) | Kernel object | M_BLOCKS / N_BLOCKS / K_MICRO |
|---|---|---|
| `v/k/q_matmul_seg`, `gg/ug_matmul_seg` | `bf16_gemm_pythoc_M8_N16_K4_AT_bf16out_s64_512_64_256_64_512.o` | 8 / 16 / 4 |
| `og/dg_matmul_seg` | `bf16_gemm_pythoc_M8_N8_K4_AT_bf16out_s64_512_64_256_64_512.o` | 8 / 8 / 4 |

Devices that share a kernel differ at the dispatch / host-arg /
shim-channel level. See `builders/o_ffn.py::_emit_{og,gg,ug,dg}_matmul_seg`
and `builders/rms_gemms_rope.py::_emit_matmul_device` for the
per-device emit shapes.

## Notes

- `Makefile` auto-points `PEANO_INSTALL_DIR` at the AIR-tree pip
  `llvm-aie` (commit `5ed1593`); the pythoc-tree in-tree `llvm-aie`
  (commit `55604435`) crashes in `InterBlockScheduling::emitLoopRemarks`
  on the RoPE-K core. Override with `PYTHOC_LLAMA_PEANO=<path>`.
- The default decode-time attention runs on CPU. NPU decode-attention
  is not implemented in this tree (the AIR reference also keeps it on
  CPU — LQ=1 makes dispatch overhead beat the NPU GEMV).

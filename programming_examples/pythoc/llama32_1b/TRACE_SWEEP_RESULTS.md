# Decode matvec trace sweep — cycle-state results

AIE hardware-trace sweep of the decode matvec sub-devices on NPU2,
instrumented with `make trace` / `trace_sweep.py` and aggregated by
`trace_summary.py`. Metrics are **cycles-in-state as % of the trace span**
(not event counts), one compute tile (col 0, row 2) per device, 48 launches
(2 tokens × 16 layers + warmup), `trace_size = 8 MiB`.

Latest sweep: **2026-06-03**, on the current packed defaults. The original
2026-05-28 device-packing-OFF baseline is preserved at the bottom.

## Metric key
- **vec_util** — vector unit active (`INSTR_VECTOR`)
- **lock_stall** — core stalled on a lock (waiting for a DMA buffer)
- **starv0 / starv1** — DMA stream starvation on S2MM ch0 (X / output) and
  ch1 (weights): the stream had no data to deliver
- **dma_in1_eff** — DMA weight-channel efficiency (ch1)
- **span** — total traced cycles across the 48 launches

## Current device layout (packed defaults)

Device-packing is **default-on**. The decode FFN/O-proj path emits packed
`aie.device`s instead of one device per matvec:

| pack mode | path | devices | notes |
|---|---|---|---|
| `d1d3d4` | **bf16 default** | 4: `rm_rms_seg`, `d1_og_a1_pack`, `d3_gg_ug_sw_pack`, `d4_dg_a2_pack` | RMSNorm is its own device |
| `d1d3d4_rms` | **AWQ default** | 3: `d1_og_a1_pack`, `d3_gg_ug_sw_pack`, `d4_dg_a2_pack` | air's 3-device fold: `rm_rms` eliminated, RMS folded into the d3 gate/up matvec prologue |

Each packed device hosts an 8-column matvec herd (cols 0–7); `d3` adds a third
compute row (rows 2–4) for the fused gate/up/swiglu. We trace the
representative tile `(0,2)` per device. Targets live in
`trace_sweep_d1d3d4_rms.targets`.

## End-to-end decode throughput

Median of 3 NPU2 runs (`d1d3d4_rms`, instruct model). AWQ A/B is the
`awq_mv` inline win (commit `053b75e8`):

| path | ms/token | tok/s |
|---|---:|---:|
| AWQ `awq_mv=.o` (baseline) | ~79 | ~12.7 |
| **AWQ `awq_mv=.ll` (inlined, shipped)** | **~69.7** | **~14.3** |

## BF16 — `d1d3d4` (4-device default)

| sub-device | role | span (cy) | vec_util | lock_stall | starv0 (X) | starv1 (W) | dma_in1_eff |
|---|---|---:|---:|---:|---:|---:|---:|
| `d3_gg_ug_sw_pack` | FFN gate+up+swiglu | 126,458,573 | 1.48% | 90.3% | 32.7% | 80.0% | 19.9% |
| `d4_dg_a2_pack` | FFN down (K=8192) | 67,407,935 | 2.45% | 77.3% | 47.8% | 61.9% | 37.7% |
| `d1_og_a1_pack` | O-proj + attn | 16,911,085 | 2.78% | 81.8% | 58.4% | 62.2% | 37.4% |
| `rm_rms_seg` | RMSNorm | 1,400,745 | 1.01% | 24.5% | 96.4% | 96.4% | 1.8% |

## BF16 — `d1d3d4_rms` (3-device fold)

| sub-device | role | span (cy) | vec_util | lock_stall | starv0 (X) | starv1 (W) | dma_in1_eff |
|---|---|---:|---:|---:|---:|---:|---:|
| `d3_gg_ug_sw_pack` | gate+up+swiglu **+ RMS** | 126,519,841 | 1.49% | 89.6% | 98.0% | 79.9% | 19.9% |
| `d4_dg_a2_pack` | FFN down (K=8192) | 67,533,732 | 2.44% | 77.4% | 47.9% | 62.0% | 37.6% |
| `d1_og_a1_pack` | O-proj + attn | 16,787,631 | 2.79% | 81.7% | 58.8% | 62.2% | 37.6% |

**The RMS fold is free on the critical path.** Folding the RMSNorm prologue
into d3 adds ~0 cycles to d3's span (126.5M either way) — the prologue is
trivial next to the weight-DMA-bound matvec — while eliminating the separate
`rm_rms_seg` device (1.4M cy + one device dispatch per token). d3's
`starv0=98%` is expected, not a regression: the fused device receives the
pre-norm res1 + ffn_norm_w as a single packed `[2,K]` burst on the X channel
(RMS computed on-tile), so that S2MM channel idles after its one delivery.

## AWQ — `d1d3d4_rms` (3-device fold, packed uint4 weights)

| sub-device | role | span (cy) | vec_util | lock_stall | starv0 (X) | starv1 (W) | dma_in1_eff |
|---|---|---:|---:|---:|---:|---:|---:|
| `d3_gg_ug_sw_pack` | gate+up+swiglu **+ RMS** | 71,231,208 | 14.8% | 60.5% | 97.9% | 63.6% | 12.9% |
| `d4_dg_a2_pack` | FFN down (K=8192) | 46,325,698 | 22.2% | 46.8% | 7.0% | 20.4% | 41.4% |
| `d1_og_a1_pack` | O-proj + attn | 10,292,753 | 25.6% | 33.9% | 14.9% | 38.0% | 29.9% |

## Headline

**BF16 decode is weight-DMA bound; AWQ shifts it toward compute.**

- BF16 matvecs idle on the per-column weight DMA: vec_util ~1.5–2.8%,
  lock_stall ~77–90%, weight-stream starvation ~62–80%. The fused `d3` is the
  decode bottleneck (126M cy span).
- AWQ packs weights to uint4 (~¼ the bytes), so every device's span drops
  ~40–45% (d3 126→71M, d4 67→46M, d1 17→10M) and lock_stall falls hard
  (d1 82→34%). vec_util jumps ~10× (1.5–2.8% → 15–26%) because the cores now
  spend real cycles on the uint4→bf16 dequant.
- This is exactly why **inlining `awq_mv` was a ~12% win and inlining the bf16
  matvec was neutral**: the dequant sits on the critical path, while the bf16
  multiply is hidden behind weight DMA. (commit `053b75e8`)
- `d3` remains the bottleneck even for AWQ (71M cy, 60% lock_stall). Remaining
  headroom is in weight delivery, not compute.

## Reproduce

```bash
# BF16, 3-device fold (works with trace_sweep.py):
PYTHOC_LLAMA_O_GEMV_FFN_PACK_MODE=d1d3d4_rms \
  python3 trace_sweep.py --targets-file trace_sweep_d1d3d4_rms.targets \
    --out-dir build_peano/trace_sweep/d1d3d4_rms_3dev \
    --n-tokens 2 --weights synthetic
python3 trace_summary.py build_peano/trace_sweep/d1d3d4_rms_3dev

# AWQ: trace_sweep.py's `--quant awq-emulate` falls through to bf16 in the
# Makefile (only QUANT=awq is special-cased), so drive `make trace` directly
# with real repacked weights, one target per device, then summarize:
for dev in d1_og_a1_pack d3_gg_ug_sw_pack d4_dg_a2_pack; do
  make trace KERNEL=o_gemv_ffn_awq SUBDEVICE=$dev COL=0 ROW=2 \
    QUANT=awq AWQ_WEIGHTS=/path/to/awq_repacked WEIGHTS=hf N_TOKENS=2
  cp -r build_peano/decode_kernel_cache/trace \
    build_peano/trace_sweep/awq_d1d3d4_rms_3dev/o_gemv_ffn_awq__${dev}__0_2
done
python3 trace_summary.py build_peano/trace_sweep/awq_d1d3d4_rms_3dev
```

## Raw artifacts
- Per-sweep aggregates: `<sweep_dir>/summary_metrics.{csv,md}` under
  `build_peano/trace_sweep/{d1d3d4,d1d3d4_rms_3dev,awq_d1d3d4_rms_3dev}/`.
- Per-target trace JSON: `<sweep_dir>/<target_slug>/trace.json` (+ `meta.json`).
- Graphs: `tools/plot_trace_metrics.py` → cycle-state bar chart; upstream
  `basic/event_trace/visualize_trace.py` → per-event timeline PNG.

---

## Historical baseline — 2026-05-28 (device-packing OFF)

Captured with `PYTHOC_LLAMA_*_PACK_MODE=none` (each matvec its own
`aie.device` at col 0, row 2), post clang-exact `reduce_add` fix. Kept for
comparison against the packed defaults above.

End-to-end (`make run`, base model, `N_TOKENS=64`, "The capital of France is"):
BF16 11.32 tok/s, AWQ (packed uint4) 12.93 tok/s.

### BF16 matvecs (`mv_pythoc.o`)

| sub-device | role | K | span (cy) | vec_util | lock_stall | starv1 (W) | dma_in1_eff |
|---|---|---|---:|---:|---:|---:|---:|
| `v_matvec_bf16_0` | QKV (V) | 2048 | 4,601,214 | 2.5% | 76.2% | 64.0% | 34.8% |
| `og_matvec_bf16_0` | O-proj | 2048 | 16,533,216 | 2.8% | 73.5% | 61.4% | 38.3% |
| `gg_matvec_bf16_0` | FFN gate | 2048 | 65,727,996 | 2.8% | 73.3% | 61.6% | 38.4% |
| `dg_matvec_bf16_0` | FFN down | 8192 | 60,965,768 | 2.7% | 74.9% | 55.7% | 42.7% |

### AWQ matvecs (`awq_mv_pythoc.o`)

| sub-device | role | K | span (cy) | vec_util | lock_stall | starv1 (W) | dma_in1_eff |
|---|---|---|---:|---:|---:|---:|---:|
| `og_awq_matvec_0` | O-proj | 2048 | 12,878,867 | 20.4% | 25.5% | 30.3% | 30.0% |
| `gg_awq_matvec_0` | FFN gate | 2048 | 50,163,613 | 21.0% | 23.5% | 27.7% | 32.5% |
| `dg_awq_matvec_0` | FFN down | 8192 | 54,795,766 | 18.8% | 34.8% | 27.6% | 30.7% |

Note: `v_matvec_awq_bf16_0` (rms_gemv_rope_awq, K=2048 QKV) segfaults under
trace instrumentation (`make trace` Error 139) — a pre-existing trace-infra
limitation on that sub-device.

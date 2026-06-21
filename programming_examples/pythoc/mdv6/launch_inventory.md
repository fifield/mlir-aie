# MDV6 warm-frame launch inventory

Regenerated **2026-06-20** against the **current model defaults**: rn3 chain on
(raster path), weight-replay (wr2), both BFP576 convs, **and the new vectorized
host pack/unpack (`MDV6_VEC_PACK`, default on)** — all default-on (no env vars
needed). Measured with `test_full_model_mc.py --profile 8` (1 cold + 7 warm),
warm-frame average, accuracy PASS (`max_class_diff 0.1423`, `max_vector_diff
0.0312`).

> **What changed since the 2026-06-12 regen.** Two things moved the numbers:
> (1) the **vectorized host pack/unpack** landed (commit `663caa40d`) — it
> replaced the per-tile torch/Python packing + reassembly with numpy
> sliding-window gather + reshape, cutting the host plumbing inside
> `launch_gap` by ~68%; (2) the profiler now **counts the rn3 chain** (it
> dispatches via `ResidentXCLBinRunner`, previously profiler-blind). So
> `launches/frame` went 84 → **98** (84 wrapped conv/gemm + 14 chain) and the
> chain's ~63 ms of NPU kernel time moved out of the inter-launch gap and into
> `npu_run`. The old "84 launches / 479 ms / launch_gap 192 ms" line is not
> comparable bucket-for-bucket because of (2); the real, like-for-like win is
> the host-plumbing collapse described below.

## Headline

| metric | value |
|--|--|
| **warm latency** | **403 ms / frame** (warm range 368–501; steady ~388, inner-fwd ~307) |
| **throughput** | **2.48 fps** |
| **launches / frame** | **98** (84 wrapped + 14 rn3-chain) |
| NPU-active fraction | 55.1% (222 ms, incl. 62 ms chain) |
| host dispatch plumbing (`launch_gap`) | 49.5 ms (12.3%), **505 µs/call** |

The model is now **NPU-dispatch + setup bound**, no longer host-plumbing bound.
The vectorization drove `launch_gap` host plumbing from ~158 ms → ~50 ms
(**−68%**); per-launch host glue is down from ~2.3 ms to **~0.5 ms/call**.

> Note: BFP convs + weight-replay are ON but contribute ~0 vs the emulated
> baseline — `npu_run` is flat with them on/off. Default-on because they're
> correct and harmless, not because they move the wall.

## Where the 403 ms goes (warm-frame buckets)

| bucket | ms | % | what |
|--|--|--|--|
| **npu_run** | **221.8** | **55.1%** | NPU active — wrapped conv/gemm (~158) + **chain.kernel 61.9** |
| **pre_post** | **118.7** | **29.5%** | pre-first-launch **96** + post-last-launch **22**. The 96 ms is the harness's per-frame `gc.collect()` (measurement artifact — inner forward timer is ~307 ms) |
| **launch_gap** | **49.5** | **12.3%** | per-launch Python/pyxrt plumbing (505 µs/call × 98) — was ~158 ms pre-vectorization |
| cpu_layers | 12.0 | 3.0% | CPU-resident (Detection, AvgPool, Upsample) |
| numpy | 0.4 | 0.1% | np.concatenate — **was 9.1 ms** (−95%, vectorization removed per-tile concats) |
| pack / fuse | 0.3 | 0.1% | weight repack / fuse_bn cache hits |
| **TOTAL** | **402.9** | 100% | |

## Per-layer NPU time (warm-frame avg, sorted)

The 222 ms `npu_run` by layer (calls/frame × µs/call). NPU compute is unchanged
by the vectorization (it's host-side) — these match prior regens.

| layer | ms | calls | µs/call |
|--|--|--|--|
| **chain.kernel** (rn3) | **61.85** | 14 | 4418.1 |
| mc_re4_c3       | 18.18 | 4 | 4544.4 |
| mc_ftconv1      | 17.80 | 4 | 4450.7 |
| mc_re6_c3       | 17.45 | 6 | 2908.9 |
| mc_elan_c3      | 12.07 | 2 | 6034.6 |
| mc_ftconv0      |  7.92 | 1 | 7915.7 |
| gemm_re6_rn1    |  7.81 | 6 | 1301.1 |
| mc_re8_c3       |  6.93 | 4 | 1732.7 |
| gemm_re4_c4     |  6.74 | 6 | 1123.5 |
| mc_aconv5       |  5.91 | 2 | 2957.1 |
| gemm_re6_c4     |  5.45 | 4 | 1362.0 |
| gemm_re8_rn1    |  5.32 | 4 | 1330.2 |
| gemm_elan_c4    |  5.17 | 6 |  861.4 |
| gemm_re4_rn1    |  5.14 | 4 | 1285.9 |
| gemm_re8_c1     |  4.86 | 6 |  809.8 |
| gemm_re6_rnm    |  4.61 | 6 |  768.5 |
| gemm_elan_c1    |  4.32 | 5 |  863.8 |
| gemm_re8_c4     |  3.82 | 3 | 1271.9 |
| mc_aconv3       |  2.96 | 1 | 2962.7 |
| mc_aconv7       |  2.86 | 1 | 2855.4 |
| gemm_re15_c1    |  2.78 | 2 | 1390.3 |
| mc_aconv16      |  2.38 | 1 | 2382.1 |
| gemm_re4_c1     |  2.10 | 2 | 1050.0 |
| gemm_re12_c1    |  1.71 | 1 | 1712.9 |
| mc_aconv19      |  1.63 | 1 | 1628.4 |
| gemm_re18_c1    |  1.24 | 1 | 1235.4 |
| gemm_re6_c1     |  1.12 | 1 | 1118.4 |

Chain host overhead (now visible): chain.host_write 0.68 ms (14), chain.host_read
0.52 ms (14) — the chain is **97% NPU kernel**, not host.
CPU-resident: cpu.Detection 6.91 ms (3), cpu.AvgPool2d 4.45 ms (5),
cpu.Upsample 0.61 ms (2). Host: np.concatenate 0.42 ms (22 calls/frame, was 1342).

## Takeaways

1. **Host plumbing is no longer the bottleneck.** The vectorized pack/unpack cut
   `launch_gap` host glue ~68% (158 → 50 ms) and `np.concatenate` 95% (9.1 →
   0.4 ms, 1342 → 22 calls). `launch_gap` is now ~50 ms (505 µs/call), near the
   irreducible BO-sync floor — further dispatch-count reduction barely moves the
   wall (confirmed: packed-GEMM −9 launches = wall-flat; both gated off).
2. **The wall is now NPU + setup bound.** `npu_run` 222 ms (55%) splits into the
   wrapped big convs (~158 ms — mc_re4_c3 / ftconv1 / re6_c3 / elan_c3 / ftconv0
   dominate) and the rn3 **chain.kernel 62 ms** (14 launches @ 4.4 ms, mostly
   on-device dispatch floor on tiny 40×40 stacks). These are *kernel/compute*
   levers, not dispatch.
3. **`pre_post` is mostly a measurement artifact.** Of the 119 ms, ~96 ms is the
   profile harness calling `gc.collect()` once per frame; the inner forward timer
   is ~307 ms. In a real streaming pipeline this overhead would not recur
   per-frame.

## Config / reproduce
```
# defaults — no env vars needed (vectorized pack on):
python test_full_model_mc.py --profile 8
# legacy per-tile packing (pre-optimization) for A/B:
MDV6_VEC_PACK=0 python test_full_model_mc.py --profile 8
# emulated-conv baseline:
MDV6_CONV1_BFP=0 MDV6_CONV2_BFP=0 MDV6_WTREPLAY2=0 python test_full_model_mc.py --profile 8
```

Optional opt-in dispatch reductions (correct but not wall wins on this HW; kept
default-off — see `OPTIMIZATION_NOTES.md`):
```
MDV6_USE_PACKED_GEMM=1   # -9 launches, wall-flat; sits at the ~29 hw-context ceiling
MDV6_USE_RNMCHAIN=1      # -6 launches but forces re6 onto the +52%/call geo chain -> net regression
```

### Methodology notes
- Per-layer NPU time is attributed via `mcr._CURRENT_LAYER` in the `_wrap_xrt`
  hook; the rn3 chain is attributed via a `ResidentXCLBinRunner.run` hook
  (`chain.*` sub-buckets) added 2026-06-20.
- `pre`/`post` split and the `chain.*` buckets are emitted by the current
  `profile_harness.py`. `MDV6_SYNC_PROF=1` additionally dumps a per-bucket
  copy_in/sync_in/copy_out/sync_out/start/wait breakdown at exit.

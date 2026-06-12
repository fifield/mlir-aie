# MDV6 warm-frame launch inventory

Regenerated 2026-06-12 against the **current model defaults**: rn3 chain on,
**raster path**, **weight-replay (wr2)**, and **both BFP576 convs** — all now
default-on (no env vars needed; disable with `MDV6_RN3RASTER=0` /
`MDV6_WTREPLAY2=0` / `MDV6_CONV{1,2}_BFP=0`). Measured with
`test_full_model_mc.py --profile 6` (1 cold + 5 warm), warm-frame average,
accuracy PASS (`max_class_diff 0.1423`).

## Headline

| metric | value |
|--|--|
| **warm latency** | **479 ms / frame** (warm range 455–533) |
| **throughput** | **2.09 fps** |
| **launches / frame** | **84** |
| NPU-active fraction | 33.1% (158 ms) |
| overhead fraction | 63.0% (launch_gap + pre_post) |

The model is **dispatch/host-bound, not compute-bound.** The largest single
cost is per-launch Python/pyxrt plumbing (`launch_gap`), ~2.3 ms/launch.

> Note: BFP convs + weight-replay are ON here but contribute ~0 vs the emulated
> baseline — `npu_run` is 158 ms with them on, 159 ms with them off (the conv
> compute / weight DMA they shrink are a small slice of `npu_run`, which is itself
> a minority of the frame). See [the BFP conv-mmul lever note]. They're default-on
> because they're correct, slightly more accurate, and harmless — not because they
> move the wall.

## Where the 479 ms goes (warm-frame buckets)

| bucket | ms | % | what |
|--|--|--|--|
| **launch_gap** | **192.5** | **40.2%** | per-launch Python/pyxrt plumbing (2291 µs/call × 84) |
| pre_post | 109.3 | 22.8% | pre-first-launch + post-last-launch (setup, last layer) |
| **npu_run** | **158.3** | **33.1%** | NPU active (DefaultNPURuntime.run + merged-ELF xrt.run) |
| cpu_layers | 9.4 | 2.0% | CPU-resident (RepConv, Detection, AvgPool, Upsample) |
| numpy | 9.1 | 1.9% | np.concatenate (host weight assembly) |
| pack | 0.2 | 0.0% | weight repacking (cache hits after warmup) |
| fuse | 0.2 | 0.1% | fuse_bn cache lookups |
| iron_alloc | 0.0 | 0.0% | XRT buffer creation |
| **TOTAL** | **479.1** | 100% | |

## Per-layer NPU time (warm-frame avg, sorted)

The 158 ms `npu_run` by layer (calls/frame × µs/call):

| layer | ms | calls | µs/call |
|--|--|--|--|
| mc_re4_c3       | 18.21 | 4 | 4551.3 |
| mc_ftconv1      | 17.69 | 4 | 4423.4 |
| mc_re6_c3       | 17.44 | 6 | 2907.2 |
| mc_elan_c3      | 12.10 | 2 | 6049.3 |
| gemm_re6_rn1    |  7.81 | 6 | 1301.7 |
| mc_ftconv0      |  7.80 | 1 | 7798.8 |
| mc_re8_c3       |  6.86 | 4 | 1716.0 |
| gemm_re4_c4     |  6.73 | 6 | 1121.7 |
| mc_aconv5       |  5.91 | 2 | 2957.2 |
| gemm_re6_c4     |  5.42 | 4 | 1355.9 |
| gemm_re8_rn1    |  5.32 | 4 | 1331.1 |
| gemm_elan_c4    |  5.18 | 6 |  863.4 |
| gemm_re4_rn1    |  5.18 | 4 | 1293.9 |
| gemm_re8_c1     |  4.81 | 6 |  801.1 |
| gemm_re6_rnm    |  4.57 | 6 |  760.9 |
| gemm_elan_c1    |  4.29 | 5 |  858.8 |
| gemm_re8_c4     |  3.80 | 3 | 1265.0 |
| mc_aconv3       |  3.13 | 1 | 3128.3 |
| mc_aconv7       |  2.86 | 1 | 2855.9 |
| gemm_re15_c1    |  2.77 | 2 | 1386.5 |
| gemm_re4_c1     |  2.40 | 2 | 1200.5 |
| mc_aconv16      |  2.37 | 1 | 2369.4 |
| gemm_re12_c1    |  1.68 | 1 | 1683.7 |
| mc_aconv19      |  1.63 | 1 | 1627.8 |
| gemm_re18_c1    |  1.25 | 1 | 1253.3 |
| gemm_re6_c1     |  1.13 | 1 | 1132.5 |

CPU-resident (off the NPU): cpu.Detection 5.61 ms (3), cpu.AvgPool2d 3.31 ms (5),
cpu.Upsample 0.53 ms (2). Host: np.concatenate 9.11 ms (1342 calls/frame).

## Takeaways

1. **`launch_gap` (192 ms, 40%) + `pre_post` (109 ms, 23%) = 63% overhead.** The
   single lever that moves the wall is cutting per-launch dispatch (2.3 ms/call ×
   84) or **fusing the 84 launches into fewer**. Everything else is rounding error.
2. **NPU compute is 33% (158 ms)** and **flat regardless of BFP/replay** — the
   conv-compute and weight-DMA optimizations are real but target a slice too small
   to matter while the frame is dispatch-bound.
3. **Heaviest NPU launches:** the `mc_*_c3` 3x3 convs (mc_re4_c3 / mc_ftconv1 /
   mc_re6_c3 / mc_elan_c3, ~3-6 ms/call) dominate `npu_run`; the rn3 gemm/chain
   launches are ~1-1.5 ms each. If/when dispatch overhead is addressed, the c3
   convs (and reducing their call counts: mc_re6_c3 runs 6×/frame) are the
   compute-side lever.

## Config / reproduce
```
# defaults — no env vars needed:
python test_full_model_mc.py --profile 6
# emulated baseline for A/B:
MDV6_CONV1_BFP=0 MDV6_CONV2_BFP=0 MDV6_WTREPLAY2=0 python test_full_model_mc.py --profile 6
```

### Methodology vs the old (Jun-10) doc
The old version listed all 92 launches individually with resident `bo_key`s and a
`merged`/`resident` path column. The current `profile_harness.py` attributes NPU
time **per-layer** (via `mcr._CURRENT_LAYER`), not per-launch with bo_key, so this
regen is layer-level. To restore exact per-launch rows, add a `(bo_key, path, ms)`
append in the `_wrap_xrt` / `_wrap_run` hooks of `profile_harness.py`.

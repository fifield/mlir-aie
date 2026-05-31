# Phase C Plan — On-chip pipelining / cross-layer fusion

Status: step A landed (8649037dc). Tracks the work after Phase A (5f7460cba).

## Progress

- **Step A: RN1 pair fusion** — done. Pair ELFs cover every rn1 dispatch in
  the model (re4/re6/re8). Wall **1375 → 1330 ms (-3.3%)**, launches
  **368 → 354**, launch_gap **237 → 203 ms**, NPU **973 → 966 ms**, PASS
  bytewise. The chain_links infrastructure is now validated end-to-end.

## Baseline (post-Phase A, post-xrt.run hook)

Profile harness `--profile 3`, warm-frame average, `USE_REGIME_XCLBINS=1 USE_REGIME_KBLOCKED=1 USE_MERGED_KERNELS=1`:

```
wall      1375 ms
npu_run    973 ms   70.8%   real xrt.run time across 368 dispatches
launch_gap 237 ms   17.2%   644 µs per dispatch
cpu_layers  32 ms    2.4%   RepConv + Detection + AvgPool + Upsample
pre_post   111 ms    8.0%
```

Per-layer Pareto (top 13 = 74% of NPU):

| Layer | ms | calls | µs/call | shape |
|---|---|---|---|---|
| mc_re6_rn3 | 73.8 | 54 | 1366 | 3x3 96→16 (40×40) |
| mc_re8_rn3 | 67.8 | 48 | 1413 | 3x3 64→16 (20×20) |
| mc_re4_c3 | 56.0 | 16 | 3498 | 3x3 64→16 (80×80, ppc=2) |
| mc_re4_rn3 | 55.8 | 12 | 4650 | 3x3 32→32 (80×80, ppc=4) |
| mc_re6_c3 | 55.1 | 36 | 1531 | 3x3 96→16 (40×40) |
| gemm_elan_c1 | 46.8 | 5 | 9355 | 1x1 64→64 (160×160) |
| mc_ftconv1 | 46.0 | 4 | 11495 | 3x3 32→16 stem |
| mc_ftconv0 | 45.3 | 1 | 45276 | 3x3 8→32 stem (640→320) |
| mc_aconv3 | 45.2 | 8 | 5649 | 3x3 64→16 (80×80, s=2) |
| gemm_re6_rn1 | 45.0 | 12 | 3751 | 1x1 96→48 (40×40) |
| gemm_elan_c4 | 42.7 | 6 | 7119 | 1x1 128→64 (160×160) |
| mc_aconv16 | 41.3 | 12 | 3443 | 3x3 64→8 (40×40) |
| mc_elan_c3 | 37.0 | 2 | 18483 | 3x3 32→32 (160×160) |

## Phase C scope reality

The original pitch ("1358 ms → <500 ms via on-chip pipelining") oversold the
upside. The 973 ms `npu_run` is real on-NPU activity — kernel compute + DDR
DMA traffic — not idle plumbing. Eliminating launches saves at most the 237 ms
launch_gap, and only if every saved launch's host overhead disappears (it
won't; xrt.run overhead per call is mostly unavoidable).

Realistic ceiling without rewriting kernels: ~150–200 ms wall savings via
launch fusion. To get below ~1.0 s wall we'd also need memtile-resident
chaining (intermediate buffers stay in L2 between kernels) — that means a
single `aie.device` containing chained workers, not multiple sub-devices
glued together by a dispatcher. Bigger surface area, multi-session work.

## re6 structure

`run_re_mc(rep_elan6, ...)` for H=W=40 input does:

```
c1   = mc_re6_c1(inp)                     # 1x1 192→192,  1 call
x1, x2 = split(c1)                         # host
x3rn = run_rn_mc(layer.conv2[0], x2, ...)  # RepNCSP, see below
x3   = mc_re6_c3(x3rn)                     # 3x3 96→16
x4rn = run_rn_mc(layer.conv3[0], x3, ...)  # RepNCSP
x4   = mc_re6_c3(x4rn)                     # 3x3 96→16
out  = mc_re6_c4(cat([x1, x2, x3, x4]))    # 1x1 384→192
```

`run_rn_mc(rep_ncsp, inp, ...)` does:

```
x1_rn   = mc_re6_rn1(inp, repncsp.conv1)  # 1x1 96→48
current = x1_rn
for bn_block in repncsp.bottleneck:        # re6 has N=3 inner blocks
    residual    = current.clone()          # host
    repconv_out = CPU bn_block.conv1(current)  # ← CPU RepConv (~600 µs)
    conv2_out   = mc_re6_rn3(repconv_out, bn_block.conv2)  # 3x3 48→16
    current     = residual + conv2_out      # host (when residual)
x2_rn  = mc_re6_rn1(inp, repncsp.conv2)    # 1x1 96→48 (same shape, diff weights)
out_rn = mc_re6_rnm(cat([current, x2_rn]), repncsp.conv3)  # 1x1 96→96
```

So one re6 frame fires ~14 NPU calls plus 3 CPU RepConvs per inner block.

## Phase C fusion candidates, ordered by ROI vs effort

### A. RN1 pair — DONE (8649037dc, 2026-05-31)
Fuse the two `mc_rn1` calls in `run_rn_mc` (conv1 + conv2 on the same `inp`).
- **Predicted**: ~7 ms wall.
- **Measured**: -45 ms wall (-3.3%), -14 launches, -34 ms launch_gap, -7 ms NPU.
  Better than predicted because per-call host overhead also dropped (575 vs
  644 µs/call) and one xrt.run.wait covers two ops instead of two waits per
  pair.
- Outputs: `merged_gemm_t{tile}_ic{ic}_oc{oc}_p{ppc}_pair_x1.elf` for the
  three rn1 shapes; `run_gemm_pair_mc` finds them by shape and falls back
  to two sequential calls when no pair ELF exists.

### B. C3 pair (medium effort)
`x3 = mc_re6_c3(x3rn)` and `x4 = mc_re6_c3(x4rn)` are called with different
inputs but identical kernel shape. Can't share input, but COULD share weights
between them IF both use the same conv weights (they don't — `layer.conv2[1]`
vs `layer.conv3[1]`). So no useful sharing.

### C. rep_elan-wide outer fusion (medium effort)
Fuse `c1 + first_rn1 + outer-c3` into one ELF with chained sub-devices via
intermediate BOs in DDR (not memtile yet). The host stops touching the
intermediate buffers (`x1`, `x2`, `x3rn`'s output, etc.). Saves ~5 host
launches per re_mc call × 3 (re4/6/8) + neck calls.
- **Effort**: medium. Need a merged ELF with 4-6 sub-devices and the dispatcher
  threading intermediate BOs between them. No memtile L2 sharing yet.
- **Savings**: ~15 launches per frame × ~700 µs = ~10 ms wall. NPU time
  unchanged.
- **Risk**: increased dispatcher complexity; need a one-shot test rig.

### D. Move RepConv to NPU — INFRASTRUCTURE LANDED, NET LOSS WITHOUT KERNEL FUSION

**What was tried (2026-05-31):** added `fuse_repconv()` to
_full_model_helpers/elan_test_tiled.py that folds the two parallel BN+Conv
branches of a RepConv into a single 3x3 conv with bias. The reparameterized
weights drop straight into the existing mc_*_rn3 kernel (bn_w=1, bn_b=B_fused).
Replaced `CPU bn_block.conv1(current)` in run_rn_mc with
`rt(mc_rn3, ..., fuse_repconv(bn_block.conv1))`.

**Result:** PASS bytewise (max_class_diff=0.219 vs 0.221 baseline; tiny
diff is bf16 rounding from the weight fusion, not a correctness regression).

**Perf:** wall **1330 → 1582 ms (+252 ms regression)**. Each bottleneck now
fires 2 mc_rn3 NPU calls instead of 1 — added 114 dispatches/frame ×
~1.4 ms NPU each ≈ +160 ms NPU + +30 ms launch_gap. Only recovers ~26 ms
of cpu.RepConv. **Net loss ~250 ms.**

**Reverted on the model path**, but fuse_repconv stays as committed
future-use infrastructure.

**Why simple port is a regression:** the math was wrong in the
pre-implementation plan. I assumed RepConv-on-NPU was incrementally
~free; it's actually a full mc_*_rn3 dispatch (1.4 ms NPU + 700 µs host
plumbing), which costs significantly more than the 600 µs CPU it
replaces.

**Real Phase D win requires kernel-level fusion:** a NEW 3x3+3x3+add+SiLU
kernel that does the bottleneck's two convs back-to-back with the
intermediate staying in memtile L2 (no DDR roundtrip, no host
patch-extract between them). Approximate target: one such fused-kernel
call replaces RepConv + mc_rn3 (currently 600 µs CPU + 1.4 ms NPU = 2.0
ms/iteration) with ~1.5 ms / iteration → ~50 ms wall saved across 102
bottleneck iterations in re6+re8.

**Effort to do it right:** new IRON kernel + memtile-resident dataflow
+ matching dispatch. Multi-session.

A shared-BO chain (à la step A) does NOT work for two rn3 calls back
to back: sub0's output is in core-packed format, sub1's input needs
patch-extracted format. The host has to do the reshape between them,
which makes BO-sharing pointless.

### E. Memtile-resident chaining (the original Phase C pitch)
One `aie.device` containing chained workers for, say, the whole rep_elan
backbone. Intermediates stay in memtile L2. Cuts DDR bandwidth, hides DMA
under compute.
- **Effort**: 1–2 weeks. New IRON kernel design.
- **Savings**: potentially 200–400 ms wall.
- **Risk**: high. Many things have to land right.

## Proposed sequence

1. ~~**A: RN1 pair fusion**~~ — DONE (-45 ms wall).
2. ~~**D: RepConv-on-NPU port**~~ — Tried + reverted. Infrastructure
   (fuse_repconv) committed for future use. Naive port was a +252 ms wall
   regression because the new mc_rn3 dispatches cost more than the CPU
   RepConv they replaced. Real win requires the fused 3x3+3x3 kernel
   in step E.
3. **C: rep_elan outer fusion** (next, smaller surface). c1 + outer-c3 +
   outer-c3 + c4 in `run_re_mc` — four convs called back-to-back with
   different intermediates. Different host plumbing pattern from A; ROI
   bounded by remaining 200 ms launch_gap.
4. **E: memtile-resident chaining / fused multi-conv kernel**. The real
   prize. RepConv lesson confirms: kernel-level fusion (intermediate in
   memtile L2, no host reshape) is the only way to compound conv
   dispatches without a regression. Multi-session, new IRON design.

## Why this differs from the pre-Phase-A pitch

The pre-Phase-A reading of "Phase C unlocks <500 ms" was based on the
profile-harness showing `launch_gap` as the bottleneck (since NPU time wasn't
attributed at the time). With the xrt.run hook in place, we now see 70% of
wall is real NPU compute + DDR DMA. Phase C as defined (cross-layer launch
fusion) saves at most the 17% launch_gap; the rest of the win needs E (or
faster kernels).

## Open questions

- Is the rn3 (~141 ms across re6+re8) actually compute-bound or DMA-bound? HW
  trace would answer this but is currently blocked by mlir-aie's single-
  column trace routing (see `conv/trace_ftconv0.py` — code is there but
  aiecc fails to route trace alongside mc_ftconv0's saturated col-0 streams).
- Is RepConv worth keeping on CPU as a baseline? Per call ~600 µs CPU. If we
  port to NPU, NPU dispatch overhead might be similar — only the dispatch
  fusion makes it worthwhile.

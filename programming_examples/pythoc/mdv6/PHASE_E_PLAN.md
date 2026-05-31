# Phase E Plan — Fused conv1+conv2 in RepNCSP bottleneck

Status: scoping (correctness analysis surfaced a wrong v0 — see below).
Phase D landed (e8233a0db) and parked +252 ms wall regression on the
run_rn_mc bottleneck path. Phase E's job is to recover that regression and
ideally drop below the pre-Phase-A baseline (1405 ms wall).

## Where the regression lives

Phase D moved RepConv from CPU (~600 µs/iter) to NPU (~1.4 ms/iter), making
the bottleneck loop fire **two** mc_rn3 dispatches per iteration instead of
one. 102 iterations × 1 extra NPU call ≈ +160 ms NPU + +30 ms launch_gap.

Per-layer cost after Phase D:

| Layer | ms | calls | µs/call |
|---|---|---|---|
| mc_re6_rn3 | 144.7 | 108 | 1340 |
| mc_re8_rn3 | 133.3 |  96 | 1388 |
| mc_re4_rn3 | 110.9 |  24 | 4623 |
| **Total**  | **389** | **228** | — |

## Compute vs DMA: rn3 is DMA-bound

Peak AIE2P bfp16 = 1.28 TOPS/tile × 32 tiles = 41 TOPS array peak.
Single rn3 MAC count (re6): 40 × 40 × 48 × 48 × 9 = 33.2 MMACs.
Compute time ceiling: 33.2e6 / 41e12 = **0.8 µs** of pure compute.
Measured: 1340 µs/call → **<0.1% compute utilization**. DMA-bound.

That's the lever. Keep the intermediate in memtile L2 (or at minimum
share a DDR BO without host roundtrip), and conv2's input DMA largely
overlaps with conv1's compute → significant NPU savings possible.

## The data-dependency wrinkle

The two convs in the bottleneck are **strictly sequential**:

```python
repconv_out = rt(mc_rn3, ..., current, ...)   # conv1
conv2_out   = rt(mc_rn3, ..., repconv_out, ...) # conv2 reads conv1's output
```

That kills the trivial "two-sub merged ELF with no chain" pattern that
worked for RN1 — the RN1 pair shared an *input* BO (one common input,
two convs in parallel). Here conv2's input *is* conv1's output, so a
shared output→input BO has to either match formats (it doesn't —
conv1 emits tile-format, conv2 wants patch-format with halo) or do an
in-merged-ELF reformat.

This means:
- Pure dispatcher-only fusion (no kernel/DMA changes) **doesn't work**.
- Every viable v0 requires the intermediate to be reformatted on-chip
  between the two sub-devices. That reformat lives in memtile DMA
  (TensorAccessPattern) or in a glue kernel — either way, real surgery.

## Implementation options (revised)

### v0 — Single-device chained conv1+conv2 with intermediate in DDR

One `aie.device` containing two stacks of conv workers. Conv1 stack
writes intermediate **back to DDR** in a layout conv2 can consume; conv2
stack reads it. Same xrt.run launches both stacks.

- **What's new**: a single device with two `multicore_conv` stacks back-
  to-back, and DMA patterns that emit/consume the intermediate without a
  host roundtrip. The conv1 output → conv2 input format conversion lives
  in memtile TAP between the two stacks.
- **Savings**: 1 dispatch per iter saved → -54 ms launch_gap.
  Probably no NPU compute savings (intermediate still DDR-bound).
- **Wall**: 1582 → ~1530 (estimate).
- **Effort**: medium. New IRON device combining two stacks; memtile TAP
  for patch-extraction-with-halo from HWC layout in DDR.
- **Risk**: medium. DMA patterns for halo-extraction inside an
  ObjectFifo are not used anywhere else in mdv6 yet.

### v1 — Memtile-resident intermediate

Same as v0 but conv1 writes intermediate to memtile L2 ObjectFifo
(not DDR). Conv2 reads from memtile. DDR traffic for the intermediate
disappears.

- **Savings**: v0's launch saving + ~40-80 ms NPU (intermediate DMA
  eliminated, since rn3 is DMA-bound).
- **Wall**: 1582 → ~1450-1500.
- **Effort**: high. Per-shape L1/L2 budget:
  - re8 (20×20×64): intermediate 51 KB, easy.
  - re6 (40×40×48): intermediate 154 KB, 30% of 512 KB memtile, OK.
  - re4 (80×80×32): intermediate 410 KB, fills a memtile (need to split
    across columns or stream).
- **Risk**: medium-high. Memtile-resident intermediate is a new pattern.

### v2 — Plus residual + SiLU epilogue in conv2 kernel

v1 plus `conv3x3_fused_packed_bf16.cc` modified to accept a residual
addend and add it post-SiLU. Removes the host residual add and one
extra DDR pass.

- **Savings**: small over v1 (~5-10 ms; residual.add is cheap on CPU).
- **Effort**: medium. Kernel change + extra dispatch arg.

## What we actually do tonight

Given:
- v0 still requires significant new IRON DMA infrastructure.
- v1 is the path to real wall savings.
- The user explicitly accepted the +252 ms Phase D regression to set up
  Phase E's host structure.

**Recommended ordering** (revised from initial plan):

1. **Spike a v1 prototype for re8** (smallest shape). Two conv stacks in
   one `aie.device`, intermediate held in memtile L2. Standalone build,
   bytewise compared against two sequential mc_re8_rn3 calls.
2. **If v1/re8 PASSes bytewise and wall improves** when wired into the
   model, scale to re6 then re4.
3. **Skip v0 entirely** — DDR-bounce intermediate has all of v1's
   complexity (memtile TAP, dual-stack device) without v1's NPU savings.

## Key contract details for the prototype

- **rn3 input format**: patches with halo (`(tile_h+2k) × (tile_w+2k) × C`
  per patch). Patches stored consecutively: `[core][slot][patch_bytes]`.
- **rn3 output format**: tile-format (no halo, `tile_h × tile_w × oc_block`),
  same `[core][slot][...]` layout.
- **Conv1 → conv2 transition**: in the v1 prototype, conv1 writes to a
  memtile-resident HWC tensor (40×40×48 for re6). Conv2 reads from same
  memtile with halo-extracted TAP per (tr, tc, core, slot). The memtile
  is the rendezvous point.
- **Weights**: separate, two weight BOs (different conv weights). Phase D
  already provides fuse_repconv weights for conv1 and fuse_bn weights for
  conv2 in the right packed layout.
- **Residual**: stays on host for v1 (cheap; ~100 µs total).

## Open questions for design phase

- Can a single `aie.device` host TWO `multicore_conv` invocations in one
  Program? The existing aie2_multicore.py builds one stack at a time —
  need to refactor it to accept two configs and emit a combined Program.
- The memtile intermediate ObjectFifo: depth=2 ping-pong with HWC layout,
  or one big buffer with TAP-driven access? IRON's ObjectFifo supports
  TAPs on consumer/producer sides.
- L1 footprint per core: conv1 worker + conv2 worker can't both live on
  the same core (one stack per core). 32 cores split 16+16 between
  stacks; each stack does half the spatial tiles.
- For re4 with 80×80 spatial and ppc=4 (the active variant), the existing
  rn3 already uses ppc=4 to keep 32-core utilization. With 16+16 cores
  per stack, ppc effectively doubles → check L1 fit.

## Out of scope

- Replacing the rn3 kernel.cc itself (besides v2's residual epilogue if
  we get there).
- Outer rep_elan fusion (PHASE_C_PLAN.md option C).
- Cross-bottleneck fusion (every iteration is its own fused dispatch).

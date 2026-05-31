# Phase E bottleneck model — where wall time actually goes

Pulled together to answer: **what's the optimal cores-per-layer, and where
could multi-residency help?** Short answer: per-dispatch overhead dominates,
not compute or DMA. That changes Phase E priorities.

## Per-layer cost decomposition (warm-frame profile, Phase D active)

Every layer runs on 32 cores. `us/call` is total wall time per xrt.run.
`comp_us` and `dma_us` are theoretical floors at bf16 peak (5.12 TFLOPS
array; 50 GB/s aggregate DDR BW). `ovhd` is what's left after subtracting
launch_gap (600 µs/call Python plumbing) and the max of compute/DMA.

```
Layer                  us/call  calls   tot_ms    mac_pc   GMAC/s  comp_us   dma_us     ovhd
-----------------------------------------------------------------------------------------------
mc_re6_rn3                1340    108      145      0.31      0.2      0.1      1.8      738
mc_re8_rn3                1388     96      133      0.15      0.1      0.1      0.9      787
mc_re4_rn3                4623     24      111      2.46      0.5      1.0     21.8     4001
mc_re4_c3                 3498     16       56     14.75      4.2      5.8     81.9     2816
mc_re6_c3                 1531     36       55      3.69      2.4      1.4     19.1      912
mc_aconv3                 5649      8       45     11.06      2.0      4.3     55.3     4994
mc_aconv5                 1224     24       29      1.84      1.5      0.7      8.9      615
mc_aconv7                  911     32       29      0.92      1.0      0.4      4.4      307
mc_aconv16                3443     12       41      1.84      0.5      0.7      9.2     2834
mc_aconv19                 917     16       15      0.69      0.8      0.3      3.3      314
mc_elan_c3               18483      2       37    117.96      6.4     46.1   1048.6    16834
mc_ftconv0               45276      1       45    235.93      5.2     92.2   2097.2    42579
mc_ftconv1               11495      4       46     29.49      2.6     11.5    163.8    10731
-----------------------------------------------------------------------------------------------
```

**Numbers that matter:**
- Frame's MAC-only work at 5.12 TFLOPS bf16 peak: **~7.5 ms** of pure compute.
- Frame's DMA-only work at 50 GB/s aggregate: **~30 ms** (DMA-bound bound).
- Measured layer-time total: **788 ms**.
- **Unaccounted "overhead": ~552 ms.**

The PHASE_E_PLAN.md "rn3 is DMA-bound" claim was wrong. rn3 layers actually
move ~140 KB per call which is ~3 µs at modest DDR rates — but each call
takes ~740 µs of NPU-active time. The bottleneck is per-dispatch overhead
(kernel start/stop, lock synchronization, DMA descriptor setup, memtile
broadcast serialization), not bandwidth.

## What this means for core count

**Reducing cores per layer would hurt.** Per-call NPU time has a large
fixed component that doesn't scale with core count. At 16 cores instead
of 32 for the same layer:
- Per-call NPU time ≈ unchanged (overhead dominates)
- Output per call halves → need 2× more calls
- Total wall ≈ 2× worse

Going the other way (more cores than 32) isn't an option on AIE2P — the
array tops at 32 compute tiles.

## Where multi-residency *could* help

The 740 µs/call NPU overhead is a per-xrt.run cost. If two INDEPENDENT
layers shared the array on disjoint column sets, each at 16 cores in 4
columns, the dispatch overhead could potentially be amortized. Rough
modeling for two re6_rn3-sized layers sharing the array:

| Pattern | NPU-active per call | Total NPU |
|---|---|---|
| Serial: layer A then layer B | 740 + 740 µs | 1480 µs |
| Co-resident on disjoint columns | ~900 µs (shared startup) | 900 µs |
| **Savings** | — | **~580 µs per pair** |

102 bottleneck iterations × 580 µs = ~60 ms wall savings — *but only if
the two layers are independent*. The Phase D bottleneck loop has a strict
data dependency (conv2 reads conv1's output) so multi-residency doesn't
apply there.

**Where it DOES apply** in the current graph:
- The RN1 pair (already merged via Phase C step A — sharing input, not
  parallel execution; but the -7 ms NPU we measured hints at exactly
  this dispatch-overhead amortization).
- Outer rep_elan structure: `mc_re6_c3` (x2) followed by `mc_re6_c4`.
  The two c3 calls are independent (different inputs) — could co-reside.
- Across-image batching (frame-level pipeline parallelism). Each frame
  uses 4 columns; two frames in flight share the array. Big design
  change, not a tweak.

## The real lever: smarter DMA, not runtime loops

To eliminate the per-dispatch NPU overhead we need **one xrt.run per
layer-call** (or close to it). IRON's `range_(...)` constructs are *not*
runtime loops — the compiler unrolls them statically into a fixed
sequence of DMA descriptors. You can't have data-dependent iteration in
the runtime sequence. So "make the loop bigger" really means **unroll
more iterations at compile time, with each iteration setting up DMA
descriptors for its own slice of input/weight/output via 4D
TensorAccessPattern**.

The existing `patches_per_core` (`ppc`) mechanism already does this for
the spatial dimension. The kernel `core_fn` contains:

```python
for _ in range_(patches_per_core):   # unrolled at compile time
    elem_in = of_in.acquire(1)        # ping-pong slot
    elem_out = of_out.acquire(1)
    kern(elem_in, elem_wt, elem_out, ...)
    of_in.release(1)
    of_out.release(1)
```

With ppc=4, the body unrolls to 4 sequential kernel calls. The memtile
DMA pattern (compile-time, configured at runtime via TAP) walks through
4 strided slices of the input BO and serves each as a patch. **One
xrt.run, 4× the spatial work** — same L1 footprint because the patches
pipeline through the same ping-pong slots.

**The missing piece is OCB unrolling.** Currently OCB is a host-Python
loop (different weights per OCB → separate xrt.run per OCB). It should
move into the runtime sequence as a compile-time-unrolled outer loop
that:

- Reads from a bigger weight BO containing `[ocb_0_w, ocb_1_w, ...]`
  concatenated; memtile strides by `weight_slot_size` per OCB iteration.
- Writes to a bigger output BO containing `[ocb_0_out, ocb_1_out, ...]`;
  memtile strides by `output_slot_size` per OCB iteration.
- Reuses the same input BO across OCBs (input is OCB-invariant — one
  fill at start, drained once at end).

For re6_rn3 today (1 OCB unrolled, 4 spatial calls × 3 OCBs = 12 xrt.runs)
this collapses to **1 xrt.run per layer-call**.

## What bounds "work per xrt.run"

Not loop depth — that's compile-time unrolled. The actual constraints:

1. **Memtile BD budget**: 48 BDs across 12 DMA channels. Each unrolled
   iteration emits its own weight-fill / output-drain BDs.
2. **Shim DMA BD budget**: 16 BDs per channel. Limits distinct strided
   patterns per direction.
3. **L1 ping-pong slots**: depth × buffer_size per core. Independent of
   unroll depth — ping-pong slots are reused across iterations.
4. **Compile time + ELF size**: roughly linear in total unrolled
   iteration count.

For re6_rn3 collapsing all 12 calls into 1 xrt.run:
- 32 cores × 4 spatial batches × 3 OCBs × 2 (in+out) ≈ 768 BD-uses
  distributed across 8 shims + 8 memtiles.
- Tight but should fit. Worth a build attempt before assuming.

## Modeled savings

| Lever | Mechanism | Wall savings |
|---|---|---|
| OCB unroll on rn3 layers (collapse 12 calls → 1) | -11 dispatches × (600 µs launch_gap + ~700 µs NPU ovhd) × 9 layer-calls | **~110 ms** on re6_rn3 alone |
| Same on re8_rn3 (8 calls → 1) | -7 dispatches × ~1300 µs × ~12 layer-calls | **~100 ms** on re8_rn3 |
| Same on re4_rn3 (already ppc=4; ~3 calls → 1 per layer-call) | -2 × ~1300 µs × ~8 layer-calls | **~20 ms** |
| **Total OCB-unroll** | — | **~230 ms wall** |

This is far bigger than the original v1 (memtile-resident intermediate)
estimate of 90-130 ms. And the implementation is pure DMA descriptor
expansion — no new device topology, no kernel.cc changes, no fused
conv1+conv2 design surface. The bottleneck conv1+conv2 fusion that
Phase E originally targeted falls out for free as a special case
(2-stack version of the same unroll mechanism), but isn't necessary
to get the bulk of the win.

## Sanity check: where does the rest of the 1640 ms wall go?

```
Layer wall (sum)                     788 ms  (48%)
launch_gap (Python) ~600 µs × 470    282 ms  (17%)  — included in layer wall above
pre/post processing                  111 ms  (7%)
cpu layers (Detection, Upsample)      32 ms  (2%)
unaccounted (likely XRT BO mgmt,    ~700 ms  (43%)
  context switching, frame
  prep/teardown)
```

The "unaccounted" 700 ms is the real puzzle. Likely candidates:
- Per-merged-ELF context activation (each fused-kernel switch).
- Buffer allocation/sync overhead between layers.
- Python-side data marshalling not captured in launch_gap.

Worth a separate investigation pass — there might be lower-hanging
fruit there than Phase E.

## Recommendations

In rough ROI order:

1. **Phase E pivots to OCB-unroll**: extend `aie2_multicore.py` to emit
   a runtime sequence with the OCB loop unrolled at compile time, using
   strided TAPs against concatenated weight/output BOs. Pure DMA
   descriptor expansion — no new device topology, no kernel.cc change.
   Modeled **~230 ms wall savings** across rn3 layers. Validate BD
   budget fits before assuming.

2. **Investigate the "unaccounted 700 ms"** (separate dig). Profile
   harness could be extended to bucket time outside layer functions.
   Could yield 100-300 ms wall savings with no kernel changes — and may
   surface that some of it is BO mgmt that disappears with OCB-unroll
   anyway.

3. **Apply the same unroll to spatial batches** if BD budget permits.
   For rn3 at ppc=1 with 4 spatial calls per OCB, this would unroll the
   spatial loop similarly. Smaller incremental win on top of OCB-unroll
   (most dispatch overhead is already amortized).

4. **Multi-residency for outer-c3 pair** (independent c3 calls in
   run_re_mc). ~10-20 ms savings, new design surface. Skip unless #1
   leaves obvious headroom.

5. **Conv1+conv2 bottleneck fusion** falls out as a special case of #1
   once OCB-unroll is in place — the same runtime-sequence machinery
   that unrolls OCBs can chain two convs. Modest additional savings
   once OCB-unroll already collapsed the dispatches.

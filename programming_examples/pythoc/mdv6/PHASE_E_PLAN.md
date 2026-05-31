# Phase E Plan — OCB-unroll for per-dispatch overhead

Status: re-scoped. Original Phase E plan targeted fused conv1+conv2 in
the bottleneck loop with a memtile-resident intermediate. After
modeling per-layer cost (see PHASE_E_BOTTLENECK_MODEL.md), that design
attacks the wrong bottleneck — rn3 layers are NOT DMA-bound, they're
**per-dispatch-overhead-bound**. The new direction is **OCB-unroll**:
collapse the host-side OCB loop into the runtime sequence at compile
time, with strided memtile TAPs serving each OCB its weight/output
slice from concatenated BOs. Pure DMA descriptor expansion — no new
device topology.

## The problem (recap from BOTTLENECK_MODEL)

Per-layer cost decomposition reveals:
- Pure compute floor (bf16 peak, 5.12 TFLOPS array): ~7.5 ms/frame.
- Pure DMA floor (50 GB/s aggregate): ~30 ms/frame.
- **Per-dispatch overhead: ~552 ms/frame unaccounted** by compute or DMA.

For `mc_re6_rn3` (the largest rn3 cost): 12 xrt.run dispatches per
layer-call × 9 layer-calls = 108 dispatches/frame. Each dispatch costs
~1340 µs (~600 µs Python launch_gap + ~740 µs NPU dispatch ovhd) but
only ~2 µs of that is actual compute or DMA.

**Why 12 dispatches per layer-call?** OCB blocking (oc_block=16, total
OC=48 → 3 OCBs) × spatial batching (32 cores × ppc=1 = 32 patches/call,
need 100 patches → 4 spatial calls). The host Python loops over both
dimensions, issuing one xrt.run per (OCB, spatial_batch).

The kernel could already pipeline through many more (OCB, spatial)
combinations within ONE xrt.run if the runtime sequence's DMA
descriptors covered them — but right now the OCB loop is on the host.

## The mechanism: compile-time OCB unroll with strided TAPs

IRON's `range_(...)` in a runtime sequence is **statically unrolled by
the compiler** — it produces a fixed sequence of DMA descriptors at
build time, not data-dependent iteration. So "do more work per xrt.run"
means emitting more DMA descriptors per build, each pointing at its
own slice via 4D TensorAccessPattern.

The existing `patches_per_core` mechanism already does this for the
spatial dimension. With ppc=4, the kernel `core_fn` unrolls 4 sequential
patch invocations; the memtile DMA TAP strides through 4 slices of the
input BO. Same L1 footprint (ping-pong slots reused), 4× the spatial
work per xrt.run.

**OCB unroll is the same trick on the OC dimension:**

```python
# Sketch — new outer loop in the runtime sequence
with rt.sequence(big_in_ty, big_wt_ty, big_out_ty) as (I, W, O):
    rt.start(*workers)
    rt.set_barrier(...)

    # Input: one fill, reused across OCBs
    for col in range(n_cols):
        rt.fill(col_in_fifos[col].prod(), I, tap_in)

    # OCB loop — unrolled at compile time
    for ocb in range(n_ocb):
        # Weight TAP strides by weight_slot_size per ocb
        tap_wt = TensorAccessPattern(
            (big_wt_size,),
            offset=ocb * weight_slot_size,
            sizes=[1, weight_slot_size],
            strides=[0, 1],
        )
        for wf in wt_fifos:
            rt.fill(wf.prod(), W, tap_wt)

        # Output TAP strides by output_slot_size per ocb
        for col in range(n_cols):
            tap_out = TensorAccessPattern(
                (big_out_size,),
                offset=ocb * output_slot_size + col_offset,
                sizes=[1, col_out_size],
                strides=[0, 1],
            )
            rt.drain(col_out_fifos[col].cons(), O, tap_out, wait=True)
```

Host changes: concatenate per-OCB weights into one `big_wt`; allocate
`big_out` of size `n_ocb * single_ocb_out`; pre-fill input once;
dispatch ONE xrt.run; unpack from `big_out` by OCB stride.

## BD budget check (the real bound)

Phase E unroll depth is bounded by:
- **Memtile BDs**: 48 total / 12 channels. Each unrolled OCB iteration
  adds weight-fill + output-drain BDs.
- **Shim DMA BDs**: 16/channel. Multiple TAPs per direction stack BDs.
- **L1 ping-pong slots**: NOT affected by unroll depth (slots are
  reused across unrolled iterations).
- **Compile time / ELF size**: linear in unrolled iteration count.

For re6_rn3 collapsing 12 calls → 1: 32 cores × 4 spatial × 3 OCBs × 2
(in/out) ≈ 768 BD-uses spread across 8 shims + 8 memtiles. Tight but
should fit. **Validate empirically before committing.**

## Implementation plan

### Step 1: Single-layer prototype on re8_rn3 (smallest)

- Pick `mc_re8_rn3` (20×20, 64→64, 5 spatial calls × 4 OCBs = 20 calls
  per layer-call today — or whatever the regime config dictates).
  Actually re8_rn3 active config is tile 4×4, oc_block=16 → 25 spatial
  tiles, oc_block factor 4. ppc=1.
- Modify a copy of `aie2_multicore.py` (call it `aie2_multicore_ocb.py`)
  to accept `n_ocb` parameter and emit the unrolled runtime sequence.
- Build standalone ELF; verify aiecc can route it (BD budget check).
- Standalone bytewise test: feed concatenated weights/output buffers,
  compare to existing mc_re8_rn3 × N_OCB sequential dispatches.

### Step 2: Wire into the model

- Modify `_run_tiled_mc_inner_merged` (or a sibling function) to
  pre-concatenate per-OCB weight blocks and allocate the bigger output
  BO. The OCB loop becomes a single xrt.run + an unpack-by-stride
  step at the end.
- Register the new ELF in `_MERGED_LAYERS_ALL` with the new mode flag.
- Run profile harness; verify wall savings match model.

### Step 3: Scale to re6_rn3 and re4_rn3

- Same template, larger dimensions. re4_rn3 already uses ppc=4, so
  fewer spatial calls per OCB but more BDs per spatial call.
- BD budget will be tightest here. Worst case: re4_rn3 at ppc=4
  needs 32 cores × 4 ppc × 1 OCB × 2 = 256 BD-uses today; unrolling
  the 1 → 3 OCBs would push that to ~768. Need to check.

### Step 4 (optional): Spatial-unroll on top of OCB-unroll

If BD budget allows, also unroll the spatial dimension. For ppc=1
layers with 4 spatial calls, this gives another 4× collapse — but
most overhead is already amortized after OCB unroll, so the
additional savings are smaller.

## Modeled savings

| Layer | Calls today | Calls after | Savings/frame |
|---|---|---|---|
| mc_re6_rn3 | 108 | 9 | ~110 ms |
| mc_re8_rn3 | 96 | 12 | ~100 ms |
| mc_re4_rn3 | 24 | 8 | ~20 ms |
| **Total rn3** | **228** | **29** | **~230 ms** |

If wall savings track the model, Phase D regression (+252 ms) is fully
recovered and the model drops to ~1410 ms — back near pre-Phase-A
baseline (1405 ms) with RepConv on NPU. Whether further unroll of c3
and aconv layers extracts more depends on BD budget headroom.

## What's deferred to a later phase

- **Conv1+conv2 bottleneck fusion**: falls out as a special case of
  this mechanism (two-stack runtime sequence with chain_links between
  the stacks). Modest additional savings once OCB-unroll has already
  collapsed dispatches.
- **Memtile-resident intermediate**: not pursued — modeled savings
  (~2 ms from DMA) don't justify the new device design.
- **Multi-residency on disjoint columns**: separate design surface;
  could be combined with OCB-unroll later but not in scope.
- **Increased ppc on rn3 layers**: subsumed by OCB-unroll (which
  amortizes overhead more aggressively than ppc=2 or ppc=4).

## Open questions

- **Does aiecc route 768 BD-uses cleanly?** Only one way to find out —
  build the re8 prototype and see.
- **L1 footprint impact**: ping-pong slots are reused, so L1 should be
  flat across unrolled iterations. But the kernel's `of_wt.acquire(1)`
  + `of_wt.release(1)` pattern is once-per-core_fn currently. With
  multi-OCB unroll, the weight ObjectFifo needs to cycle through OCB
  weights — does the existing acquire/release pattern compose, or
  does it need to move inside the OCB-unrolled loop too?
- **Trace instrumentation**: still blocked by mc_ftconv0's col-0
  saturation. Modeling has to suffice for choosing parameters.

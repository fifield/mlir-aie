# SPP9 authored resident schedule

Date: 2026-09-09. Tracking: `mlir-aie-2vb.2`. This is the concrete schedule
design for [fusion milestone 2](../FUSION_PERF_PLAN.md), not an implemented
kernel, compiled full-island placement, or performance result. Default execution
is unchanged. The subsequent [gather-only proof](GATHER_VALIDATION.md) compiles
and passes exact 30/100/300-frame NPU tests with zero compute workers. That
proves the bounded gather's routing/layout, not the complete schedule below.
The subsequent [phase-A shard](PHASE_A_VALIDATION.md) validates one worker's
full-spatial projection and three pools, with all sixteen trained channel
slices tested sequentially. The [resident packet-to-gather proof](PACKET_GATHER_VALIDATION.md)
now validates sixteen-worker opaque-bit transport and four-column gather together
in one submission. Physical phase aliasing, the finite phase controller, and
numerical composition with the final projection remain unimplemented.
The next bounded step is the [one-core alias/rearm sentinel](PHASE_ALIAS_PLAN.md).

The dependency-free [model](fusion_schedule.py) checks storage accounting and
the exact gather ordering. Run from the MDV6 directory:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m sppelan.fusion_schedule
PYTHONDONTWRITEBYTECODE=1 python -m unittest sppelan.test_fusion_schedule -v
```

Six tests pass, including with `python -O`. They establish metadata and element mapping, **not** DMA
descriptor legality, buffer bank placement, routing, floating-point equivalence,
deadlock freedom, or hardware performance.

## Chosen decomposition

Actual graph from `test_full_model_mc.py`:

```text
20×20×256 -> conv1(1×1, BN, SiLU) -> f0:20×20×128
                                   -> pool5 -> f1 -> pool5 -> f2 -> pool5 -> f3
logical [f0,f1,f2,f3]:20×20×512 -> conv5(1×1, BN, SiLU) ->20×20×256
```

Use four columns, four compute workers per column: 16 workers total. This is a
placement proposal; verify the selected NPU2 device geometry before generation.
Each worker initially owns eight adjacent neck channels, including their full
20×20 spatial extent. Worker `(column,row)` owns channels
`8*(4*column+row) .. 8*(4*column+row)+7`. Pooling is channel-independent, so
there is no spatial halo traffic. Reuse these same workers for final projection,
where each owns 16 adjacent output channels. The logical frame takes one
submission and two sequential device phases, with a device-only phase barrier.

### Phase A: project, pool, store

1. Upload packed first-projection weights and fused BN scale/bias. Retain each
   worker's 256×8 weight matrix while processing the whole frame.
2. Stream 25 input stripes, each 16 consecutive HWC pixels ×256 channels.
   Multicast each stripe across all four columns and their four workers. Every
   worker projects to its eight channels and writes its complete `f0` L1 plane.
   These are linear pixel stripes, not 4×4 spatial patches: 1×1 convolution
   makes that distinction safe. No padding pixels are needed (400/16=25).
3. After all `f0` pixels exist, apply three 5×5, stride-one, padding-two pools
   locally. Retain all four planes. Each is 6.25 KiB, total 25 KiB per worker.
4. Copy all four planes to the worker's assigned slice of its column's memtile.
   Memtile layout is `[worker_row=4, pool_level=4, pixel=400, channel=8]`,
   100 KiB per column. This is on-chip L1-to-L2 traffic, not an external spill.
5. Wait for **all** four columns' stores before reusing worker feature/input/
   weight allocations. Release phase-A FIFO objects before phase-B acquisition.
   Retain the L2 features until the entire final projection completes.

The first implementation should perform all phases sequentially. Overlap is
explicitly deferred until bank assignment and ownership are demonstrated.

### Phase B: bounded gather, project, drain

Load final-projection weights: each worker owns all 512 input channels for its
16 output channels (16 KiB plus 64 bytes fused BN). For each of 25 stripes:

1. Gather the four pool levels and all 128 neck channels into a 16×512 HWC
   stripe. Concat order is **pool level first, then neck channel**; worker-major
   storage must not be mistaken for logical K order.
2. Replicate this 16 KiB stripe to each of four destination columns, then
   multicast to the four projection workers in that column. Ping-pong storage
   bounds the resident concat copy at 32 KiB per column, not 400 KiB per column.
3. Each worker runs the final 512-to-16 projection on all 16 pixels, then BN
   and SiLU once. A full-K GEMM is the initial numerical contract; cross-worker
   partial reductions are deliberately avoided.
4. Join four 16-channel worker outputs to one 16×64 column stripe; drain each
   column to its channel slice of the contiguous 20×20×256 external output.
   The egress requires strided placement (row width 64, pixel stride 256).
5. Wait for all consumers before reusing each gather slot. Drain completion
   must include all columns before the output DeviceBuffer is marked valid.

`gather_segments(stripe)` specifies 64 source segments per destination stripe:
one for each `(pool level, eight-channel shard)`. Each segment is 16 rows ×8
bf16 elements, with source row stride 8 and destination row stride 512.
The model enumerates every destination element exactly once for all 25 stripes
and checks source level, spatial position, and channel. Offsets are elements;
IR generation must convert to the appropriate API units, not assume bytes.

Do **not** allocate 64 simultaneous DMA descriptors per destination merely
because the model emits 64 logical segments. Use a bounded reusable descriptor
schedule or a gather worker and explicitly size its scratch/queue resources.
One candidate is serial source-column rounds (four source columns, 16 segments
each) with completion before reprogramming descriptors. The implemented
[gather proof](GATHER_VALIDATION.md) instead uses four-dimensional source/scatter
descriptors and a per-destination RX0→RX1→RX2→RX3→egress lock ring, with one
stripe slot and seven static BDs per memtile. No compute-tile relay is needed.
The full-island input/weight/worker-store/output routes still need a combined
resource proof: this gather already uses five of six memtile S2MM channels.
Do not add independent worker FIFOs without budgeting how their traffic is
aggregated or phase-reconfigured.

## Storage and lifetime budget

All sizes include both ping-pong slots where named. Limits below are design
assumptions of 64 KiB per compute tile and 512 KiB per memtile, to be checked
against the compiler target. They are not a claim of a bank-feasible placement.
L1 and L2 are **not** one shared capacity.

| Worst one compute worker | Phase A | Phase B |
| --- | ---: | ---: |
| Four full channel-shard features | 25 KiB | 0 |
| Ingress / gathered stripe ping-pong | 16 KiB | 32 KiB |
| Weight matrix plus two bf16 BN vectors | 4 KiB +32 B | 16 KiB +64 B |
| Output stripe ping-pong | in scratch reserve | 1 KiB |
| Stack reserve | 4 KiB | 4 KiB |
| Scratch, accumulator spills, alignment reserve | 4 KiB | 4 KiB |
| Total | 49.03125 KiB | **57.0625 KiB** |

| Worst one memtile | Phase A | Phase B |
| --- | ---: | ---: |
| Retained four-level features, four workers | 100 KiB | 100 KiB |
| Ingress / gathered stripe ping-pong | 16 KiB | 32 KiB |
| Weight staging, four workers | 16 KiB +128 B | 64 KiB +256 B |
| Output column stripe ping-pong | 0 | 4 KiB |
| Routing / alignment reserve | 16 KiB | 16 KiB |
| Total | 148.125 KiB | **216.25 KiB** |

Weights are intentionally counted in both L1 and L2 during transfer. No
cross-frame on-chip weight residency is assumed. Fused BN arrays replace the
small scalar SPP kernel's four raw BN vectors; packing must match production.

**Explicit physical aliasing is mandatory.** Simply declaring every phase-A
and phase-B L1 buffer in IRON allocates more than 64 KiB; lifetime arithmetic
does not make static buffers automatically overlap. Either partition a shared
scratch arena with safe phase barriers or demonstrate compiler-supported
allocation reuse. Check the actual map, including FIFO depths, compiler spills,
stack, alignment, banks, and adjacent-tile accesses. The 4 KiB scratch reserve
is a provisional allowance, not measured kernel memory consumption.

The external input and output each occupy 200 KiB. The model uses existing
`TensorContract` for these contiguous HWC bf16/uint16 BOs only. The output N3
has two later graph consumers (upsample and the concat feeding re21); consumer
names are descriptive aliases, not executable node bindings. Its last-use
index here is island-local, not a full-model allocator's lifetime assignment.
Internal `Reservation` records are design metadata, not supported DeviceBuffer
storage types. Do not change a contract to `storage="L2"` and expect runtime
residency to work.

## Traffic and capability accounting

Intended logical activation external traffic is 200 KiB ingress plus 200 KiB
egress. Compared with the current host-pooling split, remove the 100 KiB f0
readback and 400 KiB concatenated-feature upload. Measure actual padded DMA
separately. Weights, instructions, and on-chip traffic are not included in those
activation figures.

The gather copies 400 KiB of logical features to each of four columns: 1600 KiB
total, of which 1200 KiB comes from other columns. This is a lower-level routing
burden, not free data reuse. Multicast ingress may also cause fourfold external
reads if implemented with independent fills rather than on-chip broadcast;
report this honestly. A slower first implementation can still establish the
residency capability, but no latency benefit is predicted by this model.

## Numerical contract and code reuse

- Production GEMM is in `../kernels/rep_elan_bf16.cc`, linked by the current
  GEMM generator. `../gemm_conv1x1/gemm_conv1x1_bf16.cc` is deprecated reference.
  The full-K kernel uses `mmul<4,8,8>`, rounds the GEMM result to bf16, rounds
  scaled BN before bias, then applies the existing rational SiLU approximation.
  Preserve these boundaries for the initial island. The 16-pixel stripe and
  output shards are multiples of these microkernel dimensions.
- Current K-blocked GEMM stores **bf16** partial sums between blocks. Switching
  to uninterrupted full-K accumulation can change rounding, even if it appears
  more accurate. Compare against the route actually selected for both SPP
  projections, not only a floating-point mathematical reference.
- The small `sppelan_bf16.cc` computes scalar dot products and approximate sqrt
  from raw BN parameters. It is useful graph scaffolding, not numerically
  interchangeable with the packed production GEMM path.
- Pool with negative-infinity padding semantics. Never zero-pad SiLU features:
  valid inputs can be negative. The existing scalar pool skips invalid positions
  with a finite negative sentinel; finite normal inputs work, but the new kernel
  should specify true `-inf` padding and test all-negative borders explicitly.
- Preserve f0/f1/f2/f3 bf16 values exactly through gather. Use unique channel/
  level/pixel sentinels to catch an otherwise plausible concat-order mistake.

## Next implementation sequence and gates

Progress update: step 1's standalone gather, step 2's arithmetic shard, and
the sixteen-worker packet-to-gather transport are validated separately.
Step 2's physical aliasing is still open; use [PHASE_ALIAS_PLAN.md](PHASE_ALIAS_PLAN.md)
as the next executable cut. Step 3's complete numerical composition is not done.
The sequence below preserves the full-island acceptance requirements.

1. Inspect current IRON memtile placement, DMA/lock limits, and cross-column
   transfer mechanisms. Choose bounded gather/relay wiring and prove a
   **synthetic full-size gather-only** program routes with exact sentinels.
   Record descriptor counts and actual buffer maps per tile. This gate is now
   established by the separate gather and packet-to-gather proofs.
2. Implement explicit phase-L1 aliasing and one first-projection/pool shard,
   with full 20×20 spatial shape and eight channels. Test borders, negative
   features, and distinct weights. Verify resource usage from compiler output.
3. Connect all shards, projection weights, gather, and final projection in one
   coherent program. New full-K accumulation differences require a predeclared
   numerical tolerance and comparison with both current hybrid SPP and PyTorch.
4. Expose through the persistent executor behind an opt-in island route with
   explicit input/output DeviceBuffers. Compare ordinary host-materialized,
   external-BO-connected, and on-chip implementations on the same workload.
   Initialization and CPU reference must remain outside inference-only timing
   for every compared implementation.
5. Capability gate: one submission, one island context, both convolutions and
   all pools on device, no intermediate external activation sync/transfer;
   instrument this rather than infer it from output correctness. Run changing
   inputs and trained weights, then full-model strict numerical checks and
   30/100/300-frame stability before any default promotion.

Do not report milestone 2 complete on the strength of the schedule tests. If
routing or physical aliasing fails, update this authored model with the actual
alternative (for example, stripe-streamed gather or smaller output shards),
including its new storage and traffic costs, before implementing it.

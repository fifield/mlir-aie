# SPP9 device-resident gather proof

Date: 2026-09-09. Tracking: `mlir-aie-2vb.2.1`, within the still-incomplete
`mlir-aie-2vb.2` first-island contract task. Built on `a89803b33` plus this
increment. This is the next bounded step in [the fusion plan](../FUSION_PERF_PLAN.md).

## Outcome and limits

The full-size four-column gather compiles and passes **5, 30, 100, and 300
frames on the NPU**, checking every output bit at all four destinations. It
retains worker-major features in memtiles, performs the concat ordering and
cross-column replication entirely through DMA, and reuses one bounded stripe
slot. No compute kernel, host stripe loop, intermediate upload, or readback is
needed. This establishes the hardest data-layout/routing part of the authored
[SPP schedule](FUSION_SCHEDULE.md), not full SPP fusion.

Neither convolution nor pooling executes in this proof. The source features
are uploaded diagnostic data, not device-produced activations, and the four
replicated concat outputs are downloaded for inspection. The full model and
its default route are unchanged. No full-model speedup, full-island L1 aliasing,
cross-frame weight residency, or fused arithmetic equivalence is claimed.

## Exact ABI and DMA schedule

| External BO | Native uint16 shape | Bytes |
| --- | --- | ---: |
| Input | `[source_column=4, worker_row=4, pool_level=4, pixel=400, lane=8]` | 409600 |
| Diagnostic output | `[destination_column=4, stripe=25, pixel=16, K=512]` | 1638400 |

Logical K order is pool level, then neck channel. Neck channel is
`source_column*32 + worker_row*8 + lane`. Input is copied in its native
worker-major order; only the independent CPU reference transposes it.

[aie2_gather_probe.py](aie2_gather_probe.py) declares four source and four
destination memtiles in columns 0–3, with no compute workers:

1. Four initial shim transfers fill a 100 KiB retained source buffer per column.
2. One permanent source MM2S descriptor traverses each full buffer using
   `(size, element_stride)` dimensions
   `[(25,128), (4,12800), (4,3200), (128,1)]`. Its stream order is
   stripe, worker row, pool level, pixel/lane. It multicasts to all four columns.
3. At each destination, one S2MM descriptor per source scatters each 2048-element
   part directly into the full 8192-element stripe buffer, with dimensions
   `[(4,8), (4,128), (16,512), (8,1)]` and element offset `32*source_column`.
   The four source parts cover the stripe exactly once, without scratch copies.
4. A per-destination lock ring serializes source receivers:
   `RX0 -> RX1 -> RX2 -> RX3 -> egress -> RX0`. Only RX0 starts ready. The
   egress releases the stripe after draining all 8192 elements. Every multicast
   destination uses the same source order. A shared four-token empty counter
   would be unsafe: one receiver could consume multiple tokens and overwrite
   data belonging to another stripe.
5. Source empty/ready locks retain each full frame until all 25 parts have
   been sent. Four final shim drains collect the diagnostic outputs. The
   runtime awaits **all four outputs** before freeing input tasks and returning.

Descriptors self-loop without per-stripe host programming. Same-memtile
loopback uses matching source/destination DMA channel numbers. Remote paths
are compiler-routed through the array stream network; memtiles have no direct
east/west stream ports. Routing was accepted by the compiler and exercised on
hardware, not independently decoded from switch configuration.

An initial ObjectFifo forward/join approach was rejected by the compiler:
an intermediate FIFO could not participate in both links. The direct static
DMA program avoids that lowering restriction. `make remote` retains a small
column-0-to-column-1 compile-only cut with equal 51200-element I/O; it does not
use the full four-destination host ABI and was not separately run on hardware.

## Compiled footprint

The actual `gather_probe.mlir.prj/input_with_addresses.mlir` assigns, per memtile:

| Resource | Observed allocation |
| --- | --- |
| Retained source | byte interval `[0, 102400)` |
| One gathered stripe | byte interval `[102400, 118784)` |
| Total buffer bytes | 118784 (116 KiB) |
| Static DMA descriptors | 7, reused for every stripe/frame |
| Locks | 7: IDs 0–4, 16, 17 |
| S2MM channels | 5: channels 0–4 |
| MM2S channels | 2: source column channel and channel 4 |
| Compute workers / kernel objects | 0 / 0 |

[check_gather_ir.py](check_gather_ir.py) gates buffer types, extents, bounds,
nonoverlap, resource counts, channel allocation, and all-output completion.
It is a focused checker for this compiler's textual IR, not a general MLIR
verifier, lock-order proof, or physical-routing decoder. Five CPU mutation
tests exercise its failure paths. Twelve other CPU tests check independent
semantic/segment oracles, exhaustive four-dimensional transfer mappings,
persistent BO use, and fail-stop behavior.

The full-island authored peaks remain **58432 bytes L1 per worker** and
**221440 bytes L2 per column**. Those estimates are not proven by this smaller
DMA-only map. In particular, physically aliasing phase-A/B L1 storage and
fitting the additional worker routes/channels remain required integration gates.

## Hardware evidence

Root ran NPU jobs serially after independent source/map review, with generator,
host ABI, and artifacts frozen. Platform: Strix Halo NPU2, kernel
`6.17.0-20-generic`; installed mlir-aie under
`/home/jfifield/npu-dev-mdv6/install/mlir-aie`.

| Run | Exact result | Warm host wall, excluding first frame |
| --- | --- | ---: |
| 5 frames | PASS | 0.486 ms |
| 30 frames | PASS | 0.407 ms |
| 100 frames | PASS | 0.456 ms |
| 300 frames | PASS | 0.464 ms |

Each run is a fresh process; frames within it reuse one context and two BOs.
The five-pattern cycle is low 16 bits of source index, high 16 bits, changing
random raw bits, zero, and all-one bits. The **pair** of index frames uniquely
identifies all 204800 source elements; neither frame alone does. Random seeds
change between cycles. Every one of 819200 output elements is checked each
frame against two independent CPU oracles. Opaque uint16 patterns can encode
bf16 NaNs intentionally; a floating-point finite check would be inappropriate.

Every frame records one successful submission, one upload (409600 B), and one
download (1638400 B). No context loads, misses, or evictions occur within the
measured frame scope; the initial load/allocation is outside it. The timer
includes the host input copy, sync, submission/wait, output sync, and owned
output copy. Oracle construction/comparison is outside it. These sub-millisecond
numbers are **diagnostic host wall times, not device cycles, bandwidth, or an
SPP latency prediction**. Driver context switches and internal DMA bytes remain
unmeasured. Schedule-derived gather payload is 1600 KiB total, 1200 KiB remote.

Temporary evidence (rebuild if removed):

- Artifacts and build logs: `/tmp/mdv6-gather-direct.uiKjGP/`.
- Per-frame logs: `/tmp/mdv6-gather-{5,30,100,300}.jsonl`.
- Generator SHA256: `53080c1b1fd72565a795f17fc2e6c2bb56552b47fd952b4b4c3372e3a4369a29`.
- Xclbin SHA256: `0b359ed66d57dc816db2f31dc99f7ecd105cc3b0cba42d8b7f4b83f6aaa20730`.
- Instructions SHA256: `c0876bca359a89e818b55ff441af4e6072550d8386b14a64acf1cef7a8d9da5c`.

## Reproduce in a fresh context

```bash
source /home/jfifield/npu-dev-mdv6/env.sh
export PYTHONPATH="/home/jfifield/npu-dev-mdv6/install/mlir-aie/python${PYTHONPATH:+:$PYTHONPATH}"
cd /home/jfifield/npu-dev-mdv6/mlir-aie/programming_examples/ml/mdv6
gather_build=$(mktemp -d /tmp/mdv6-spp-gather.XXXXXX)
make -C "$gather_build" -f "$PWD/sppelan/Makefile.gather_probe" all remote
python3 -O sppelan/check_gather_ir.py --build-dir "$gather_build"
PYTHONDONTWRITEBYTECODE=1 python3 -O -m unittest \
    sppelan.test_fusion_schedule sppelan.test_gather_layout sppelan.test_gather_ir
python3 -O sppelan/test_gather_probe.py --build-dir "$gather_build" \
    --frames 30 --failure-dir "$gather_build/failures-30"
```

After success, run 100 then 300 frames serially, using distinct failure
directories. Never launch concurrent NPU jobs or modify sources/artifacts
during a hardware window. Any runtime failure or mismatch stops immediately;
there is no automatic retry/recovery. If supplied, the failure directory gets
an exclusive NPZ containing input, expected output, observed output when
available, and the error. A device timeout requires recovery before more NPU
work, not a blind retry.

[run_spp_gather_probe.lit](../run_spp_gather_probe.lit) fresh-builds the program,
checks the compiled footprint, and runs 30 exact frames under `python -O`.
Run configured hardware tests with `lit -j1`; the CPU cases are also included
in [run_host_checks.lit](../run_host_checks.lit).
Both configured lit tests passed, including a separate fresh gather build.
The complete CPU suite passes 111 tests under `python -O`. A further clean
build of both full and remote targets in `/tmp/mdv6-gather-build-gate.dc1DIa`
passed the compiled-map check. Removing the generated instruction file from
the active artifact pair (retaining a `.saved` copy) correctly triggered the
grouped-target rebuild and reproduced identical instructions.

## Next bounded implementation

Update: the arithmetic cut below is now implemented and validated in
[PHASE_A_VALIDATION.md](PHASE_A_VALIDATION.md). Its all-slice 80-case matrix
and sustained 100/300-frame gates pass. The phase-alias and combined-channel
work remains outstanding; the newer document gives the next packet-aggregation
sentinel cut. This section retains the original shard contract for traceability.

Keep this pure-data routing proof as a regression target. The next cut is a
full-spatial, eight-channel phase-A worker: trained conv1 + fused BN/SiLU,
followed by three local 5x5 pools, exporting all four feature planes for exact
inspection. Establish its numerical and physical-memory contracts before
combining 16 workers with the gather. Do not change the production route yet.

Concrete shard contract and gates (tracked as `mlir-aie-2vb.2.2`):

1. Add `aie2_phase_a_shard.py`, `phase_a_shard.cc`,
   `Makefile.phase_a_shard`, and a fail-stop hardware test. The bf16 input is
   `[400,256]`; diagnostic output is `[level4,400,8]` (25600 bytes). Process
   25 linear stripes of 16 pixels. Pool only after all f0 pixels exist.
2. Preserve the deployed first-projection reduction: the current 256→128
   SPP projection selects **KB128**, so an eight-channel shard requires two
   ordered K128 calls, including the intermediate bf16 store and last-K-only
   BN/SiLU. Reuse the corrected production kernel/literal-row helper. An
   uninterrupted K256 reduction would change rounding. Each shard weight
   chunk has 1040 logical bf16 elements; use **1056-element aligned strides**.
   Observe and record rounding mode rather than silently changing it.
3. Compare f0 to the corresponding channels of a freshly built corrected
   baseline. Then require f1–f3 to match CPU max-pooling of that exact observed
   f0. Independently test pool-only all-negative borders, corners, edge impulses,
   and every channel; padding must be negative infinity. Exercise all 16 trained
   eight-channel slices sequentially through the one-shard artifact before
   replicating workers. Add changing-input/weight tests.
4. Avoid a duplicate 25-KiB readback copy in L1. For the standalone proof,
   write all four planes into one acquired depth-one output FIFO object,
   releasing it only after pooling. Compile and inspect buffers, banks, stack,
   code size, and spills before running the device.
5. Follow with a separate sentinel proof of explicit phase aliasing. One
   candidate byte-addressed shared arena is:

   | Phase | Features / gather | Input / weights | Remaining weights / output |
   | --- | --- | --- | --- |
   | A | features `[0,25600)` | input ping-pong `[25600,41984)` | aligned weights `[41984,46208)` |
   | B | gather ping-pong `[0,32768)` | weights `[32768,49216)` | output ping-pong `[49216,50240)` |

   These are proposed offsets, not a compiled allocation. Add stack/scratch
   outside the arena. Use explicit pointer views and DMA ownership; prove all
   phase-A feature stores reach L2 before any overlapping phase-B write.
   Independent FIFO declarations do not automatically implement this aliasing.

Then demonstrate phase-A/B physical scratch reuse and a compiled channel/route
budget for worker ingress, retained-feature stores, gather-to-worker multicast,
weights, and final output joins. The current gather consumes five of six S2MM
channels per memtile: adding independent worker FIFOs without planning their
join or phase reuse will not fit automatically. Only after those gates should
the two convolutions and pools be connected into one full SPP submission.
This integration proof is tracked as `mlir-aie-2vb.2.3`.

Full-island acceptance remains: one submission, device-produced intermediates
never read back or spilled externally, 200 KiB logical activation ingress and
200 KiB egress, trained-weight/boundary/changing-frame validation, and measured
traffic. The expected removal of the existing 100 KiB f0 readback and 400 KiB
concat upload is a future result, not accomplished by this diagnostic export.

# Bounded packet stripe join validation

Date: 2026-09-09. Code/planning baseline: `f4d4479f4`.
This is the second packet milestone after
[full-plane aggregation](PACKET_AGGREGATE_VALIDATION.md).

## Outcome and limits

A single submission now processes 25 stripes using reusable one-stripe
memtile slots and permanent DMA chains. Four workers return tagged payloads
through one receive channel, which scatters their channels into pixel-major
output. Independent 6-, 30-, 100- and 300-frame hardware runs passed every bit.

This proves a bounded, phase-B-shaped transport schedule in one column.
It does not perform SPP projection, pooling, cross-column gather, phase-A/B
buffer aliasing, or the complete fusion controller. No full-model dispatch
reduction or performance improvement is claimed for this diagnostic.
See [FUSION_SCHEDULE.md](FUSION_SCHEDULE.md) for the intended larger dataflow.

## Frozen ABI and independent oracle

Runtime arguments are `[input, output]`, both opaque uint16 arrays:

| Argument | Shape | Elements / bytes |
| --- | --- | ---: |
| Input | `[stripe25,worker4,pixel16,channel16]` | 25,600 / 51,200 |
| Output | `[stripe25,pixel16,channel64]` | 25,600 / 51,200 |

For stripe `s`, worker `w`, pixel `p` and local channel `c`:

```text
output[s,p,16*w+c] = input[s,w,p,c] XOR (0x1111*(w+1))
```

The host oracle applies worker tags and independently transposes the input.
All bits are checked, including infinity/NaN encodings and signed-zero bits.
XOR proves traversal through the correct worker; this is not bf16 arithmetic.
The source array remains unchanged, and returned output owns its host storage.

The test cycle contains unique global indices, stripe/worker identity tags,
seeded random words, zeros, all-one bits and opaque edge patterns.
Random seeds change between cycles. CPU checks prove all 25,600 source and
destination coordinates are covered once, including first/last stripes and
pixel/channel boundaries. Reordered workers, stale stripes and worker bypass
are explicitly rejected.

## Repeated stripe protocol

One host input DMA streams the entire frame, not 25 Python submissions.
For each stripe, the memtile receives 1,024 words into its source slot.
It sends 256-word payloads to workers in order **3,2,1,0**.
Each worker tags and retains its payload until granted permission to return.
Grants and aggregate reception proceed in order **0,1,2,3**.

Receive BDs scatter each worker's 16 channels using dimensions
`[(16,64),(16,1)]` and offset `16*worker`, covering the 1,024-word output
stripe exactly once. Once all four returns complete, the stripe drains to
the host output stream. Only that drain releases `stage_empty`, permitting
the source slot to accept the next stripe. The same credits and descriptors
cycle 25 times per frame and continue safely across subsequent frames.

Memtile channels retain the full-plane proof's division of responsibilities:
S2MM5 ingress, MM2S5 addressed payloads/grants, S2MM4 worker returns and
MM2S1 egress. Worker receive/send channels are S2MM1/MM2S0.
Input packet IDs are `1,2,4,8`; return IDs are `16..19`.
The runtime awaits its whole-frame output token before freeing the input task.

This repeated stripe-credit protocol is **not** a finite phase-A/phase-B
controller. It neither proves a one-time phase transition nor allows live
phase-A allocations to alias phase-B buffers without further ownership work.

## Actual allocated addresses and resources

The gate reads the compiler's addressed IR and all four worker ELFs.
Addresses below are byte offsets; ranges are half-open.

| Location / allocation | Range | Allocated bytes |
| --- | --- | ---: |
| Memtile source slot | `[0,2048)` | 2,048 |
| Memtile aggregate slot | `[65536,67584)` | 2,048 |
| Memtile grant token | `[131072,131136)` | 64 |
| Each worker stack | `[0,4096)` | 4,096 |
| Each worker payload | `[4096,4608)` | 512 |
| Each worker grant | `[16384,16448)` | 64 |

Memtile allocated buffers total **4,160 bytes**, but the bank-separated
address high-water mark is **131,136 bytes**. Per worker, buffers total
576 bytes plus a 4,096-byte stack; the high-water mark is **16,448 bytes**.
Do not confuse allocation sums with the occupied address span or claim the
bank gaps are automatically available for phase aliasing.

There are 14 permanent memtile BDs and 14 locks; each of four workers uses
3 BDs and 5 locks. Nothing is unrolled into 25 sets of descriptors.
Each ELF contains 672 bytes of text and zero data/BSS bytes.
The structural/resource checker reports PASS and
`physical_routing_verified=false`: it does not exhaustively verify switch
fabric routing. Exact hardware results establish this isolated path; physical
coexistence with the intended four-column aggregate/gather graph remains unproven.

## Recorded validation

| Run | Result | Warm host mean, excluding frame 0 |
| --- | --- | ---: |
| 6 frames | PASS | 0.2687 ms |
| 30 frames | PASS | 0.2119 ms |
| 100 frames | PASS | 0.1977 ms |
| 300 frames | PASS | 0.1738 ms |

Every frame recorded one completed run, one 51,200-byte upload and one
51,200-byte readback, with zero loads, cache misses or observed evictions.
There is no intermediate host transfer or per-stripe host loop.
Device cycles, device DMA bytes and driver context switches are not measured.
Times include the synchronous host wrapper's copies, transfers and launch,
but exclude initialization/oracle/validation. They are diagnostic timings,
not a matched benchmark or a prediction of full-model performance.

Eight host CPU tests and five IR/resource tests pass. The complete MDV6 CPU
suite passes all 153 tests under `python -O`. Five configured lit gates pass
serially: host checks, packet aggregation, packet stripe join, gather probe,
and phase-A shard. Stripe lit builds fresh artifacts and checks 30 frames;
the log is `lit-final.log` in the evidence directory below.
The full-model hardware test was not rerun: its route and arithmetic are unchanged.
Failures stop without retries and optionally preserve exclusive NPZ evidence.

## Reproduce

Run from MDV6, build separately, and serialize hardware tests:

```bash
source /home/jfifield/npu-dev-mdv6/env.sh
export PYTHONPATH=/home/jfifield/npu-dev-mdv6/install/mlir-aie/python:${PYTHONPATH:-}
stripe_build=$(mktemp -d /tmp/mdv6-packet-stripe.XXXXXX)
make -C "$stripe_build" -f "$PWD/sppelan/Makefile.packet_stripe_join" -j2
python sppelan/check_packet_stripe_join_ir.py --build-dir "$stripe_build"
python -m unittest sppelan.test_packet_stripe_join_host sppelan.test_packet_stripe_join_ir
python -O sppelan/test_packet_stripe_join.py --build-dir "$stripe_build" --frames 30 --failure-dir "$stripe_build/fail-30"
python -O sppelan/test_packet_stripe_join.py --build-dir "$stripe_build" --frames 100 --failure-dir "$stripe_build/fail-100"
python -O sppelan/test_packet_stripe_join.py --build-dir "$stripe_build" --frames 300 --failure-dir "$stripe_build/fail-300"
```

Evidence directory: `/tmp/mdv6-packet-stripe.slhr2K/`, containing
`resource-gate.json` and `stripe-{6,30,100,300}.jsonl`.
Preserve evidence and coordinate recovery after a timeout; do not retry a
poisoned context. Temporary build/log paths are not durable storage.
Recorded SHA-256 values:

```text
aie2_packet_stripe_join.py bb0278efb2e3ae569c4877eadff01dc8f22e12cb0ddf8d93be766f32bc4ccf6b
packet_stripe_join.cc      c1818d753f8c8cedb14d47a8e6d568cceb742e5ce96d3af5a0e35ea7450be270
packet_stripe_join.xclbin  ac84bec952ac85cc2b52807b5e2de39fef7d48ecd40f24e1eb068fd041c8702a
packet_stripe_join.bin     f406c8fdc31d313e5b4dfd526f26c68356b91f12c7659e533efaad00c00bed30
```

## Next bounded integration: resident aggregate-to-gather handoff

Tracked as `mlir-aie-2vb.2.3.3`, under the still-incomplete phase/worker
integration task `mlir-aie-2vb.2.3`. Compose the **full-plane** aggregation
proof with the gather proof; do not substitute this stripe-shaped source ABI.
Use separate artifacts and keep all existing proofs as regressions.

1. Replicate full-plane packet workers across four columns (16 cores), with
   column-qualified symbols. Each memtile retains a 100 KiB diagnostic source,
   a 100 KiB tagged aggregate, and a 64-byte grant token.
2. Remove the old packet MM2S1 diagnostic egress entirely. Instead, gather's
   MM2S channel `col` reads that column's aggregate directly with dimensions
   `[(25,128),(4,12800),(4,3200),(128,1)]` (uint16 element units). Acquire
   `output_ready`; release `stage_empty` only after all 51,200 source elements
   have been read. This prevents the next frame from overwriting the resident
   aggregate prematurely; it need not wait for downstream host readback.
3. Keep packet S2MM5 ingress, MM2S5 payload/grants, and S2MM4 returns unchanged.
   Add gather S2MM0..3, a 16 KiB concat-stripe slot, and temporary MM2S4 host
   drain. Reserve the self-route MM2S `col` to S2MM `col`. Use an independent
   five-lock stripe receive/drain ring (IDs 14..18, only ID 14 initialized to 1)
   so a stripe cannot be overwritten while still draining.
4. Check the proposed per-memtile budget: **19 BDs, 19 locks, six S2MM and
   three MM2S channels, 221,248 buffer bytes**. This is logical endpoint
   feasibility, not proof of physical routing or bank placement. Inspect actual
   allocated ranges and all 16 ELFs before hardware. MM2S4 is a temporary
   diagnostic drain, occupying a channel intended for later activation traffic.
5. Compile packet/circuit coexistence before further implementation. Reusing
   packet IDs across four columns is unproven; inspect masks and switch routes.
   The one-column workaround does not establish combined routability.
6. Freeze input `[4,4,4,400,8]` and output `[4,25,16,512]` uint16 ABIs. The
   independent oracle applies per-worker XOR tags in native source layout,
   then performs the existing gather transpose and replication to all four
   destinations. Use four input and four output DMA tasks in one runtime
   submission; await **all outputs before freeing any input**.
7. Gate compiled resources/ownership and CPU address coverage, then serialize
   a small hardware smoke test and exact changing-frame 30/100/300 runs. Stop
   on failure and preserve evidence. Require one submission and no host sync
   between aggregation and gather, then add a fresh-build configured lit test.

This cut removes the diagnostic host round trip between aggregation and gather.
It still does not execute numerical SPP. Physical L1 phase aliasing, the finite
one-phase-A/25-phase-B/rearm controller, and integration of the validated
projection/pooling shard plus final projection remain subsequent gates.

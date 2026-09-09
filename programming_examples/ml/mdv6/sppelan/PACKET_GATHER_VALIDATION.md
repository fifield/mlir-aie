# Resident packet-to-gather transport validation

Date: 2026-09-09. Code/planning baseline: `5bc072348`.
Issue: `mlir-aie-2vb.2.3.3`.

## Status and scope

The combined four-column packet/circuit design compiles and passes its
addressed-resource checker. Independent 6-, 30-, 100- and 300-frame hardware
runs pass every output bit. Nine host and six IR/resource CPU tests pass;
all six configured lit gates pass. This establishes the combined resident
transport route for the tested workload, not general numerical-SPP correctness.

This cut combines [full-plane aggregation](PACKET_AGGREGATE_VALIDATION.md)
with the [gather schedule](FUSION_SCHEDULE.md), retaining their intermediate
aggregate on the memtiles. It does not substitute the
[stripe-join proof's](PACKET_STRIPE_JOIN_VALIDATION.md) smaller input ABI.
It implements opaque XOR-tagged transport, not numerical SPPELAN.
No finite phase-A/B controller, L1 phase aliasing or full-model optimization
is established by compilation or by the transport oracle alone.

## Frozen ABI and exact oracle

One runtime call uses arguments `[input, output]`:

| Argument | uint16 layout | Elements / bytes |
| --- | --- | ---: |
| Input | `[source4,worker4,level4,pixel400,lane8]` | 204,800 / 409,600 |
| Output | `[destination4,stripe25,pixel16,channel512]` | 819,200 / 1,638,400 |

The independent coordinate oracle is:

```text
O[d,s,p,128*l + 8*(4*c+w) + a]
    = I[c,w,l,16*s+p,a] XOR (0x1111*(w+1))
```

Thus logical K order is pool level, source column, worker, then channel lane.
All four destinations receive identical complete concat features.
Worker tags prove compute-worker traversal; source identity is independently
checked and must not be inferred merely from identical destination replicas.
Opaque signed-zero, infinity and NaN encodings are valid transport words.

The runtime uploads the native input layout without host tagging or repacking
an intermediate. CPU tagging/gathering exists only to construct expected output.
The wrapper keeps one context and two BOs, returns an owned host output copy,
and preserves the caller's input. There is no Python loop over device stripes.

## Resident aggregation-to-gather protocol

Each source column first receives its 100 KiB native payload. Addressed packet
delivery visits workers 3,2,1,0; grants permit returns in order 0,1,2,3.
The four tagged worker payloads fill that column's 100 KiB aggregate.
Unlike the standalone aggregation proof, there is no aggregate host egress.

The gather sender acquires `output_ready` and reads the aggregate directly.
Its permanent source BD uses element dimensions
`[(25,128),(4,12800),(4,3200),(128,1)]`, sending all 51,200 source words.
It releases `stage_empty` only after that complete source transfer finishes.
This keeps the resident aggregate protected until it is consumed; it does
not unnecessarily tie aggregate reuse to the final downstream host readback.

At each destination, four receiver BDs scatter source contributions into
one 16 KiB concat stripe with dimensions `[(4,8),(4,128),(16,512),(8,1)]`
and offset `32*source`. A separate five-lock ring, IDs 14..18, permits
sources 0..3 in order, then drains the completed stripe before reuse.
The same descriptors process all 25 stripes; they are not unrolled 25 times.

The runtime starts four input and four output tasks. It awaits **all four**
output completion tokens before freeing any input task.
Packet IDs `1,2,4,8` and `16..19` are reused within each column's packet path.
Compilation alone does not prove every physical switch/mask behavior. Exact
changing-frame hardware tests now validate this combined bounded route;
the checker still is not an exhaustive physical switch decoder.

## Channels and actual resource map

Per memtile, packet S2MM5 ingress, MM2S5 payload/grants and S2MM4 returns
remain unchanged. Gather uses MM2S `column` and S2MM0..3; the local loopback
uses the same source/receive channel number. MM2S4 temporarily drains concat
stripes to the host. This diagnostic drain occupies a channel intended for
later activation traffic; future composition must account for that conflict.

The addressed map is identical in each of four memtiles:

| Allocation | Byte range, half-open | Bytes |
| --- | --- | ---: |
| Native source | `[0,102400)` | 102,400 |
| Tagged aggregate | `[102400,204800)` | 102,400 |
| Concat stripe | `[204800,221184)` | 16,384 |
| Grant token | `[221184,221248)` | 64 |

Allocated bytes and high-water mark are both **221,248 per memtile**.
Each of 16 workers reserves stack `[0,4096)`, planes `[4096,29696)` and
grant `[29696,29760)`: 29,760 bytes including stack. All 16 ELFs have 560-byte
text and zero data/BSS. Program text is separate from the L1 data allocation.

Per memtile: **19 permanent BDs, 19 locks, six S2MM channels and three MM2S
channels**. Each worker uses three BDs and five locks. The checker reports
24 logical circuit routes, 32 logical packet routes and four output awaits.
It reports `physical_routing_verified=false`: this is a structural/resource
gate, not an exhaustive decoder of the placed physical switch fabric.

## What host traffic this design removes

A hypothetical split aggregate-then-gather execution would read back the
409,600-byte aggregate and upload those same 409,600 bytes again for gather.
The resident connection removes that logical **819,200-byte intermediate
host round trip** and puts both transport stages in one submission.
This is a structural accounting comparison, not a measured matched baseline.
The combined diagnostic still uploads 409,600 bytes and downloads 1,638,400
bytes per frame. Historical isolated timings do not establish a speedup.

## Validation gates and recorded results

Nine host CPU tests pass: explicit coordinate oracle versus independent transpose,
paired index halves covering all 204,800 source identities in every output,
source swaps, worker/level transposes, stale/missing destinations, stripe
reordering, worker bypass, buffer persistence, failure invalidation and
exclusive failure evidence. The paired sentinel frames are necessary because
either uint16 index half alone aliases source elements.

The hardware gate requires every output bit to match, exactly one completed
run, one 409,600-byte upload, one 1,638,400-byte readback and zero load calls
per frame. Changing random inputs, constants and opaque edge words complement
the paired sentinels. Failure stops without retries and may save an exclusive
NPZ containing input, expected output, observed output when available and error.
Every recorded frame met those run/transfer requirements and also recorded
zero context-cache misses and zero evictions. Device cycles, device DMA bytes
and driver context switches remain unmeasured.

| Independent hardware run | Result | Warm host mean, excluding frame 0 |
| --- | --- | ---: |
| 6 frames | PASS | 0.5074 ms |
| 30 frames | PASS | 0.4561 ms |
| 100 frames | PASS | 0.4124 ms |
| 300 frames | PASS | 0.4172 ms |

Times include the synchronous wrapper's native input copy/upload, run and
output readback/copy. They exclude initialization, oracle construction and
validation. They are diagnostic host timings, not a matched full-model
performance comparison; no speedup is inferred from earlier isolated proofs.

Six new IR/resource tests pass, and root verified all 168 MDV6 CPU tests under
`python -O`. `lit-final.log` records all six configured gates passing serially:
phase-A shard, host checks, gather, packet stripe join, packet aggregation and
packet gather. The packet-gather lit gate builds fresh artifacts and checks
30 frames. These are scoped regression gates, not a full-model performance run.

## Reproduce from MDV6

```bash
source /home/jfifield/npu-dev-mdv6/env.sh
export PYTHONPATH=/home/jfifield/npu-dev-mdv6/install/mlir-aie/python:${PYTHONPATH:-}
packet_gather_build=$(mktemp -d /tmp/mdv6-packet-gather.XXXXXX)
make -C "$packet_gather_build" -f "$PWD/sppelan/Makefile.packet_gather" -j2
python -O sppelan/check_packet_gather_ir.py --build-dir "$packet_gather_build"
python -O -m unittest sppelan.test_packet_gather_host sppelan.test_packet_gather_ir
python -O sppelan/test_packet_gather.py --build-dir "$packet_gather_build" --frames 6 --failure-dir "$packet_gather_build/fail-6"
python -O sppelan/test_packet_gather.py --build-dir "$packet_gather_build" --frames 30 --failure-dir "$packet_gather_build/fail-30"
python -O sppelan/test_packet_gather.py --build-dir "$packet_gather_build" --frames 100 --failure-dir "$packet_gather_build/fail-100"
python -O sppelan/test_packet_gather.py --build-dir "$packet_gather_build" --frames 300 --failure-dir "$packet_gather_build/fail-300"
```

Serialize hardware access. Preserve evidence and coordinate recovery after a
timeout; never retry a poisoned context. Frozen compilation directory:
`/tmp/mdv6-packet-gather.bINPtR/` (temporary, not durable evidence storage).
Recorded hardware logs are `packet-gather-{6,30,100,300}.jsonl`; configured
lit evidence is `lit-final.log` in the same directory.
Verified SHA-256 values:

```text
aie2_packet_gather.py 8352218e9c86dd5667259911ab14a639364197b332f3cf1b3c649785a14d2480
packet_gather.cc      48e7659b3e0c88b39f51d14aecb1881a26e34febb411fab33ab716e538934539
packet_gather.xclbin  6a2831e165eb28b01e240a287302550d47960eef732aa93522c955603125596a
packet_gather.bin     c0876bca359a89e818b55ff441af4e6072550d8386b14a64acf1cef7a8d9da5c
```

## Next bounded proof

[PHASE_ALIAS_PLAN.md](PHASE_ALIAS_PLAN.md), tracked as `mlir-aie-2vb.2.3.4`,
specifies one core with explicit physical L1 overlap: one phase-A transfer,
exactly 25 phase-B transfers, then rearm. It uses a finite controller and
distinct phase/stripe sentinel tags, without numerical arithmetic.
That isolated sentinel now [passes exact sustained hardware gates](PHASE_ALIAS_VALIDATION.md),
but does not establish a full 16-worker phase barrier. The next integration
is [numerical phase A to resident gather](PHASE_A_GATHER_PLAN.md).
The current packet/gather proof does not authorize aliasing still-live buffers
or supply the missing full-island controller.

# One-column packet aggregation validation

Date: 2026-09-09. Planning/code baseline: `5c7222ac3`.
Issue: `mlir-aie-2vb.2.3.1`.

## Outcome and scope

Four compute workers successfully retain separate feature payloads and return
them, in granted order, through one shared memtile receive channel.
Independent 6-, 30-, 100- and 300-frame hardware runs passed every output bit.
The source payloads arrive in reverse worker order; output grants run forward.
Each worker applies a distinct XOR tag, ruling out a simple DMA bypass.

This completes an isolated single-column packet/ownership proof. It does not
execute SPPELAN mathematics, share allocations with phase A or B, demonstrate
coexistence with gather routing, or establish any full-model speedup.
The default full-model execution path is unchanged by this proof.

Related context: [fusion schedule](FUSION_SCHEDULE.md),
[fusion roadmap](../FUSION_PERF_PLAN.md).

## ABI and protocol

Both host arguments are contiguous `uint16[4,4,400,8]`, ordered
`[worker, pool_level, pixel, channel_lane]`: 51,200 elements / 102,400 bytes.
Runtime argument order is `[input, output]`.
Every bit pattern is valid, including signed-zero, infinity and NaN encodings;
this diagnostic performs unsigned XOR, not floating-point computation.

For worker `w`, the independent expected output is:

```text
output[w, ...] = input[w, ...] XOR (0x1111 * (w + 1))
```

Ingress first fills the full memtile source buffer. Addressed payload packets
then visit workers **3, 2, 1, 0**. After all payloads have been sent, separate
grants permit returns **0, 1, 2, 3**. Each worker retains its own four planes
and applies its tag before honoring its grant.
Receiver ownership and grant credits prevent multiple senders from racing for
the shared aggregate receive channel. The final aggregate is worker-major.
The `stage_empty` credit remains held until memtile egress finishes reading
the aggregate; the next frame cannot overwrite data still being read there.
The host separately awaits the complete external output transfer.

Relevant memtile channels are:

| Function | Channel |
| --- | --- |
| Host ingress | S2MM 5 |
| Addressed payloads and grants | MM2S 5 |
| Shared worker packet return | S2MM 4 |
| Aggregate host egress | MM2S 1 |

Worker payload/grant reception uses S2MM 1; return uses MM2S 0.
Input packet IDs are `1,2,4,8`; output IDs are `16,17,18,19`.
Packet headers are removed before writing payload buffers.

## Resource and ownership gate

The generated address/lock/descriptor map and all four ELF files pass
`check_packet_aggregate_ir.py`. Recorded gate:
`/tmp/mdv6-packet-aggregate.vZby5D/resource-gate.json`.

| Resource | Measured/configured allocation |
| --- | ---: |
| Memtiles / compute workers | 1 / 4 |
| Memtile source buffer | 102,400 bytes (100 KiB) |
| Memtile aggregate buffer | 102,400 bytes (100 KiB) |
| Grant token buffer | 64 bytes |
| Memtile allocated data span | 204,864 bytes |
| Permanent memtile BDs / locks | 14 / 14 |
| Per-worker BDs / locks | 3 / 5 |
| Per-worker L1 occupied span, including stack | 29,760 bytes |
| Per-worker stack reservation | 4,096 bytes |
| Per-worker ELF text | 560 bytes |
| Per-worker ELF data / BSS | 0 / 0 bytes |

Worker data consists of 25,600 bytes of planes and a 64-byte grant buffer,
after the stack reservation. Program text is separate from the L1 data span.
Descriptors are permanent chains, not dynamically allocated per host frame.
The host awaits the output completion token before freeing its input task.

The static checker explicitly reports `physical_routing_verified=false`:
it validates structural contracts, not the entire physical switch fabric.
Successful compilation plus exact hardware output supplies additional routing
evidence; neither substitutes for checking coexistence with a future graph.

## Validation results

| Independent hardware run | Result | Warm host mean, excluding frame 0 |
| --- | --- | ---: |
| 6 frames | PASS | 0.4299 ms |
| 30 frames | PASS | 0.3175 ms |
| 100 frames | PASS | 0.2639 ms |
| 300 frames | PASS | 0.2423 ms |

Every frame recorded one successful completed submission, one 102,400-byte
upload, one 102,400-byte readback, zero loads, zero context-cache misses and
zero observed evictions. There are no intermediate activation host transfers.
Driver context switches, device DMA bytes and device cycles remain unmeasured.

Times cover the synchronous host wrapper, including its input copy/upload,
submission and output readback/copy. They exclude initialization, CPU oracle
construction and validation. These are diagnostic host timings, **not** a
matched performance comparison or a prediction of fused-SPP latency.

Each six-frame cycle includes unique linear source indices, two independently
seeded random inputs, zeros, all-one bits and opaque edge-bit patterns.
All 51,200 source identities fit uniquely in a single uint16 sentinel.
The oracle checks every output element and its worker-specific tag.
Caller inputs are preserved. Runtime/count/mismatch failures invalidate the
probe, stop immediately without retries, and optionally save exclusive NPZ
evidence containing input, expected output, observed output when available,
and the error. Existing failure evidence is never overwritten.

CPU gates: **8 host tests + 5 IR/resource-checker tests passed**.
The complete CPU suite passes 140 tests under `python -O`. All four configured
lit gates pass: host checks, fresh packet build/run, gather, and phase-A shard.
The timing table above comes from the separate direct test invocations.

## Compiler constraints encountered

Descriptor emission had to satisfy the compiler's paired-lock constraints;
the working BDs use paired acquire/release ownership operations.
Contiguous addressed-input packet IDs produced merged-mask routing conflicts.
Failed compiler IR artifacts are `aiecc_failure_1788983118_1509986.mlir` and
`aiecc_failure_1788983167_1510780.mlir` under the build's `packet_aggregate.mlir.prj/`.
One-hot input IDs `1,2,4,8`, with return IDs `16..19`, compiled successfully.
This is a workaround for the observed compiler/router mask restriction,
not evidence of a hardware packet-routing defect.
Reduction/investigation is tracked separately as `mlir-aie-2vb.9`; packet-ID
reuse across four columns is not yet established.

## Reproduce in a fresh session

Run from the MDV6 directory. Use the installed Python modules, and build into
a fresh directory so this proof cannot overwrite production artifacts.

```bash
source /home/jfifield/npu-dev-mdv6/env.sh
export PYTHONPATH=/home/jfifield/npu-dev-mdv6/install/mlir-aie/python:${PYTHONPATH:-}
packet_build=$(mktemp -d /tmp/mdv6-packet-aggregate.XXXXXX)
make -C "$packet_build" -f "$PWD/sppelan/Makefile.packet_aggregate" -j2
python sppelan/check_packet_aggregate_ir.py --build-dir "$packet_build"
python -m unittest sppelan.test_packet_aggregate_host sppelan.test_packet_aggregate_ir
python -O sppelan/test_packet_aggregate.py --build-dir "$packet_build" --frames 30 --failure-dir "$packet_build/fail-30"
python -O sppelan/test_packet_aggregate.py --build-dir "$packet_build" --frames 100 --failure-dir "$packet_build/fail-100"
python -O sppelan/test_packet_aggregate.py --build-dir "$packet_build" --frames 300 --failure-dir "$packet_build/fail-300"
```

Run hardware serially. Stop after any timeout or runtime failure; preserve
evidence and coordinate device recovery instead of retrying a poisoned context.
Recorded logs: `packet-{6,30,100,300}.jsonl` beneath
`/tmp/mdv6-packet-aggregate.vZby5D/`. Temporary logs are not durable artifacts.

Recorded SHA-256 values:

```text
packet_aggregate.xclbin deae61e29f4fe2c7f787455dd6b67a639b90188c1ea61dcedca5d3a55c0510d3
packet_aggregate.bin    1ca9c0d0303cacf134429aad3fb07baf13108315b4ef0eff556656ac6c65faaf
aie2_packet_aggregate.py ee32ea297e1e7694c680a56ea0edd7237ab6e837f2c59fccfc9b03a866a52d67
packet_aggregate.cc     4255788bbca837cf48cd13a364cb1baa119e28a9952e0db324d28eced335d32a
```

## Next bounded integration step

The subsequent reusable per-stripe join (`mlir-aie-2vb.2.3.2`) now passes its
own compiled resource and exact 30/100/300-frame hardware gates. Its ABI is
input `[25,4,16,16]`, output `[25,16,64]` uint16 with the same worker tags.
It uses 2 KiB input/output slots and a 64-byte grant token, retaining 14 BDs
across all 25 stripes. See [its independent evidence and next handoff](PACKET_STRIPE_JOIN_VALIDATION.md).
The next integration cut is four-column full-plane aggregation feeding the
gather directly; phase-buffer aliasing and finite phase switching remain separate.

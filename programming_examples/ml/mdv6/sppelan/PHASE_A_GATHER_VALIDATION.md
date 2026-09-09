# Numerical phase A to resident gather: validation handoff

Tracked as `mlir-aie-2vb.2.4`; design is
[PHASE_A_GATHER_PLAN.md](PHASE_A_GATHER_PLAN.md).

Status: the numerical prototype is **compile-blocked by `mlir-aie-8cu`**.
Host/harness implementation is frozen and 12 optimized CPU tests pass. The
independent full-width production reference compiled successfully. Integrated
routing fails; no xclbin or integrated NPU result exists. The full
resource/instruction-binary gate and hardware campaigns remain unproven.
This milestone establishes implementation and inspected lowering, not numerical
device execution, physical routing, or a performance improvement.

## Deliverable and frozen host ABI

Produce all 16 eight-channel phase-A shards, including their three pooling
levels, into resident four-column packet aggregation and gather. The intended
route has one instruction submission with no host feature intermediate.
Artifacts are `spp_phase_a_gather.xclbin` and `spp_phase_a_gather.bin`.

Runtime arguments are `[I, W, O, M]`, all native uint16:

| Argument | Shape | Host bytes per frame |
|---|---|---:|
| I | `[400,256]` | 204800 uploaded |
| W | `[4 columns,4 workers,2 chunks,1056]` | 67584 uploaded |
| O | `[4 destinations,25 stripes,16 pixels,512 K]` | 1638400 read back |
| M | `[4 columns,4 workers,32]` | 1024 read back |

The host keeps one context and four BOs alive. Each frame requires one run,
one returned run, one completed run, zero loads, two uploads totaling 272384
bytes, and two downloads totaling 1639424 bytes. These are host sync counts.
Four shim tasks each read the same I BO: actual activation ingress is 819200
bytes, not 204800. The harness does not measure device DMA bytes or cycles.
Four replicated diagnostic outputs are not the eventual fused model traffic.

## Numerical contract and independent reference

The required production reference is
`gemm_t28_ic256_oc128_kb128_p1`: full IC256/OC128, tile28, 32 cores,
PPC1, two ordered K128 chunks. One baseline call covers all 400 pixels.
Do not substitute a single full-K accumulation or change rounding mode.

For each worker, each physical weight chunk holds 1024 blocked matrix words,
8 BN scales, 8 BN biases, and 16 zero padding words. The 1056-word stride is
2112 bytes, preserving 64-byte alignment. Intermediate bf16 rounding occurs
after the first K128 chunk; BN/SiLU applies only after the second.

The integrated kernel wrapper reuses `phase_a_shard.cc`, which preserves the
validated production projection and pooling contracts. Reusing this source
does not itself validate the new distribution schedule. Its metadata pointer
is the feature packet suffix at word 12800, never the grant buffer.

The harness requires trained `model.spp9.conv1` weights via `load_model()` and
`fuse_bn()`. Every frame contains all 16 trained eight-channel slices. Whole
shards rotate among worker positions, including their matching BN parameters,
to expose stale weights. All 16 raw variants remain alive for the baseline's
identity-keyed packed-weight cache; do not create disposable variants in-loop.

Six cases cycle: random, negative constant, spatial boundaries, weighted
K128 cancellation, zero, and alternating signed-zero inputs. Random seeds
change between cycles. Default is 30 frames; minimum is six. The 30/100/300
campaigns repeatedly exercise changing input and weight state.

Reference f0 is the fresh full-width production output, not the reused shard
kernel. Three CPU 5x5 max-pools then run sequentially with stride1/pad2.
The oracle selects the first row-major maximum, including signed-zero ties,
with negative-infinity padding. Clipped-window CPU tests independently verify
that implementation, including negative borders and corner impulses.

Gather channels are level-major: K=`128*level + 32*source + 8*worker + lane`.
Every destination must exactly match all 400 pixels and 512 channels.
All outputs must be finite bf16, M[...,0] must be floor mode 0, and every
reserved metadata word must be zero. No tolerance or checksum replaces
bitwise equality. The f0 check is production equivalence, not an independent
mathematical GEMM or PyTorch accuracy claim; pools/gather have independent
semantic oracles. Numerical phase B is absent.

## Ownership and completion contract

The proposed schedule retains full input and per-column weights on memtiles,
distributes weights once and 25 activation stripes to each worker, receives
ordered feature/metadata packets, and gathers features without host readback.
Metadata must drain before releasing the aggregate's frame credit.

The final pre-routing addressed map has the following byte intervals on every
tile. All buffer bases are 64-byte aligned and the intervals do not overlap.
This does not replace inspection of the final successfully routed artifact.

| Core allocation | Half-open byte interval |
|---|---|
| Stack | `[0,8192)` |
| Feature/metadata packet | `[8192,33856)` |
| Activation stripe | `[33856,42048)` |
| Weights | `[42048,46272)` |
| Grant | `[46272,46336)` |

| Memtile allocation | Half-open byte interval |
|---|---|
| Input | `[0,204800)` |
| Aggregate | `[204800,307456)` |
| Weights | `[307456,324352)` |
| Gather stripe | `[324352,340736)` |
| Grant token | `[340736,340800)` |

Static and runtime memtile BDs must have disjoint, channel-accessible IDs.

The final prototype uses standard distinct controller IDs: memtile 26, core
rows 2–5 respectively 27/29/30/31, and shim 15. Header-preserving control
routes have the compiler's supported `priority_route` attribute. All tasks
start before waiting. Waits drain memtile ingress first, then worker grant-RX
and packet-TX completions in worker order, followed by remaining memtile
completions and finally all eight shim outputs. All input/other task frees
remain after every wait. This ordering reduces avoidable dependency backlog;
the installed firmware's unsolicited-token capacity is not established here.
Hardware TCT-stall flags exist, so compilation alone cannot prove backpressure
safety. Never remove completion fences to make the route fit.

Audit all activation iteration addresses, finite queue counts, grant ordering,
metadata lifetime, and all eight O/M shim completions. Core/memtile device
completion tokens must fence task reuse inside the same submission.
`dma_free_task` is allocator bookkeeping, not a device fence. Arena locks and
output readback alone do not prove all finite descriptor queues retired.

## Known compiler queue defect and local workaround

`mlir-aie-96e` tracks a confirmed installed-compiler lowering bug: generic
`aie-dma-to-npu` task starts mask BD IDs with `0xf`, but NPU2 memtile start
IDs use six bits and odd channels require IDs 24–47. The compile-only
`repro_memtile_queue_id.mlir` emits 8 instead of 24 at register `0x1a065c`.
That incomplete reproducer must never run on hardware.

The example-local `start_memtile_task_6bit()` emits explicit memtile queue
writes while retaining normal descriptor configuration and task awaits.
It checks BD bank/channel parity and encodes the full BD ID, repeat count,
and token flag. Core/shim task starts retain their existing lowering.
The compiled gate must verify actual queue addresses, six-bit start IDs,
repeat/token fields, controller routes, and an exact tie to the instruction
binary. Source inspection alone is insufficient. This does not patch the
shared compiler or installed toolchain; remove the workaround only after a
fixed compiler is installed and the complete gate is rerun.

## Fail-stop behavior and CPU evidence

Twelve host CPU tests pass under `python -O`: independent all16 weight packing,
BN/padding/alignment, trained-shard permutations, independent first-max pools,
all gather channel identities, stale/axis/nonfinite mutations, each of 512
metadata words, input patterns, persistent native bindings, runtime failures,
semantic/count poisoning, strict counters/exclusive evidence, and bad ABI.

The baseline recovery loader is patched to raise before any retry, and its
run/returned/completed/load counters are checked. Integrated runtime, count,
metadata, finite-value, or bit mismatch poisons the probe and stops the run.
Failure NPZ files use exclusive creation and contain input, raw/packed weights,
and any available f0 reference, expected result, output, metadata, and error.
Startup/input-construction errors precede the per-frame evidence block; a
readback exception can prevent saving partially retrieved data. No retry,
automatic reload, or tolerance relaxation is authorized by this harness.

## Fresh production baseline provenance

Trained weights SHA-256:
`4d0af6e3d80bbbdbcc22a5ec6cf997f66e3f6062e463a04c1da34cf08f44c3ae`.

Compile-only reference root: `/tmp/mdv6-phase-a-gather-baseline.gg3LZv`.
Build reported one successful configuration, zero failures. No NPU job was
run to create this reference. Artifacts are under its `gemm/` directory:

```text
gemm_t28_ic256_oc128_kb128_p1.xclbin
SHA256 7983eb0c9cba6cfc2cc926d5f4d9cf1c8e4bc32d5fc26da9153689fa88ba5839
gemm_t28_ic256_oc128_kb128_p1.bin
SHA256 7e96d10f703ce75c74156a3d85ff92fbd6d222415cbd1761a102f4de2b1de6b8
```

## Reproduction and pending evidence

Initial integrated compile root: `/tmp/mdv6-phase-a-gather.47HmVp`.
Its addressed-map check passes: all sixteen workers have 46336 allocated
bytes including stack, with the same high-water address; all four memtiles
have 340800 allocated bytes and high-water address. The allocator falls back
from bank-aware to sequential placement; the inspected map has no overlaps.
This is memory-placement evidence, not successful routing or hardware proof.

Both initial and prioritized-control-route attempts failed the stream-switch
verifier. The first post-failure dump shows shim packet rules using invalid
NOC/PLIO source ports for completion tokens. `mlir-aie-8cu` tracks routing
feasibility: the cause may be router allocation/masking limitations, physical
oversubscription, or their interaction. Illegal emitted routes alone do not
prove a legal full route exists. Logs are `build.log`, `build-priority.log`,
and `route-debug.log`; the project retains failure IR ending
`1788988222_1591151` and `1788988561_1595268`. No failed artifact was run.
Do not bypass the verifier or remove completion fences to obtain a build.

A shared-controller-26 experiment also failed routing in
`/tmp/mdv6-phase-a-gather-shared-tct.1cBhiP` (failure ending
`1788988774_1598620`). An explicit local-hop vertical control tree failed in
`/tmp/mdv6-phase-a-gather-tree.vxh12i` (failure ending
`1788988916_1603220`): shim(0,0) North3 was used both by a circuit to East1
and a control packet route to South0. The supported fixed-connection helper
accounts for `ConnectOp`, not packet merge-tree reservations. Neither failed
experiment is the committed candidate. The final direct-route candidate is
retained separately under `/tmp/mdv6-phase-a-gather-blocked.yd0ky3`.
Its final routing failure ends `1788989255_1606909`. Frozen source SHA-256:

```text
aie2_phase_a_gather.py d6f5011207f11f6091d92b3eafb949256bd388b4a80f342ece120640a554ffb8
phase_a_gather.cc cbf91b2a77fa7e391f901f4a0c8fea3cb177e7dad37fb7ea4e28d55aad692417
```

Compile-only reductions remove selected explicit completion routes, leaving
all data routes and runtime tasks unchanged. They are **not executable**:
their completion waits intentionally have disconnected token sources.
Results in `/tmp/mdv6-routing-reductions.E1ETeh` and the exhaustive 15
column-0 subsets in `/tmp/mdv6-routing-col0-subsets.hG3E7L`:

| Retained additional completion sources | Router result |
|---|---|
| None | PASS |
| Four memtiles | PASS |
| One row-2 worker per column | PASS |
| Any single worker in column 0 | PASS |
| Five of the six column-0 worker pairs | PASS |
| Workers `(0,4)` and `(0,5)` only | FAIL: packet/circuit South2 source conflict |
| Workers `(0,2)`, `(0,3)`, `(0,5)` | PASS |
| All four workers in column 0 | FAIL: packet-mask collision |

This is not a simple route-count threshold. It points toward placement-sensitive
allocation/masking, but does not exclude a physical bottleneck in the failing
placement or prove full-design feasibility. Memory and endpoint DMA-channel
counts are necessary budgets, not a proof that shared stream-switch cuts fit.

The durable reducer preserves every byte outside removed explicit control
routes, validates the expected twenty source tiles, refuses to overwrite an
evidence directory, and runs only the routing pass. Reproduce the discriminating
pair/triple cases with a new output path:

```bash
phase_a_gather_build=/tmp/mdv6-phase-a-gather-blocked.yd0ky3
python -O sppelan/reduce_phase_a_gather_routes.py --self-test
python -O sppelan/reduce_phase_a_gather_routes.py \
  "$phase_a_gather_build/spp_phase_a_gather.mlir.prj/input_with_addresses.mlir" \
  /tmp/mdv6-routing-new-evidence --case min_pair --case passing_triple
```

The second command intentionally exits 1 if a routing case fails. Inspect
`results.json` and logs: the recorded outcome is `min_pair` FAIL and
`passing_triple` PASS. The smallest failure has memtile(0,1) South2 used
simultaneously by a circuit to DMA2 and a packet-17 rule to DMA4. Four
optimized parser/identity/selection/guard checks pass; these are additional
to the 205 unit tests. Never execute the reduced IR on hardware.
Root independently reproduced the pair/triple outcome in
`/tmp/mdv6-router-review.7iNqsV/reduced`.

From the MDV6 directory, compile into a fresh isolated root:

```bash
source /home/jfifield/npu-dev-mdv6/env.sh
export PYTHONPATH=/home/jfifield/npu-dev-mdv6/install/mlir-aie/python:${PYTHONPATH:-}
phase_a_gather_build=$(mktemp -d /tmp/mdv6-phase-a-gather.XXXXXX)
make -C "$phase_a_gather_build" -f "$PWD/sppelan/Makefile.phase_a_gather"
MDV6_BUILD_DIR="$phase_a_gather_build" python3 gemm_conv1x1/build_gemm_conv1x1.py gemm_t28_ic256_oc128_kb128_p1
python -O -m unittest sppelan.test_phase_a_gather_host
```

Nine optimized IR tests pass. After a blocked compile has emitted the addressed
map, lowering can be inspected independently without creating an executable:

```bash
aie-opt "$phase_a_gather_build/spp_phase_a_gather.mlir.prj/input_with_addresses.mlir" --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu -o "$phase_a_gather_build/spp_phase_a_gather_tasks.mlir"
aie-opt "$phase_a_gather_build/spp_phase_a_gather_tasks.mlir" --aie-dma-to-npu -o "$phase_a_gather_build/spp_phase_a_gather_runtime.mlir"
python -O sppelan/check_phase_a_gather_ir.py --build-dir "$phase_a_gather_build" --compile-only
```

Root's final component gate passes all 540 independently expected runtime
operations, including 160 descriptor writes, exact addresses/iteration/locks,
full six-bit memtile queue starts, and 56 completion waits. It also checks
the twenty static gather descriptors, terminating task chains, bounded
projection loop, and physical allocation map. `COMPONENT_PASS` explicitly
reports `hardware_approved=false`; routing, ELF sizes, actual instruction
binary and hardware execution remain unchecked. Full checker mode fails
closed without xclbin/bin/ELFs. All 205 optimized CPU regressions pass.

After routing succeeds, run the checker without `--compile-only`, inspect its
actual binary/address evidence, and freeze artifacts before any serialized NPU job.
Only after independent approval, run each campaign with a new evidence path:

```bash
python -O sppelan/test_phase_a_gather.py --build-dir "$phase_a_gather_build" --baseline-dir "$phase_a_gather_build" --frames 6 --failure-dir "$phase_a_gather_build/fail-6"
python -O sppelan/test_phase_a_gather.py --build-dir "$phase_a_gather_build" --baseline-dir "$phase_a_gather_build" --frames 30 --failure-dir "$phase_a_gather_build/fail-30"
python -O sppelan/test_phase_a_gather.py --build-dir "$phase_a_gather_build" --baseline-dir "$phase_a_gather_build" --frames 100 --failure-dir "$phase_a_gather_build/fail-100"
python -O sppelan/test_phase_a_gather.py --build-dir "$phase_a_gather_build" --baseline-dir "$phase_a_gather_build" --frames 300 --failure-dir "$phase_a_gather_build/fail-300"
```

Retain JSONL stdout and stderr. Stop at the first failure. Evidence placeholders
for root to replace from completed logs:

- Integrated executable and its hashes: **ABSENT; compile-blocked**.
- Addressed allocations and declared BD/lock/channel map: **COMPONENT_PASS**; ELF sections unbuilt.
- Queue/iteration/control-token lowering: **COMPONENT_PASS**; actual instruction-binary tie unproven.
- Exact six-frame smoke and 30/100/300-frame campaigns: **PENDING**.
- Warm diagnostic timing, measured counter records: **PENDING**.
- Complete optimized CPU regression: **205 PASS**; configured CPU lit passes,
  new hardware lit is **UNSUPPORTED**, not passed.

## Next concrete integration cut

First resolve `mlir-aie-8cu` with a reduced compile-only routing-feasibility test. Starting
from the retained addressed IR, run `aie-create-pathfinder-flows` with
`--mlir-print-ir-after-failure` and check packet-source legality and circuit
exclusivity. Check the failing placements against physical port/cut and packet
classification budgets. Establish a legal route or identify the exhausted
resource before choosing a compiler fix versus a design revision. Any
supported route or separately reviewed fix must preserve every completion fence.
Do not suppress verification or
rewrite a rejected routed artifact into an executable. The generic six-bit
queue fix (`mlir-aie-96e`) is separate; retain the local queue workaround until
the installed compiler is corrected and retested.

Then rebuild in a new directory, run the complete actual-map/runtime/binary
gate and independent route review, freeze source/artifacts, and run the serial
6/30/100/300 numerical campaigns above. The hardware lit entry
`run_spp_phase_a_gather.lit` requires `mdv6_phase_a_gather_routing`, which no
lit config currently enables. Remove that temporary feature guard only after
the routing prerequisite is resolved; its current skip is not a test pass.

After numerical phase-A/gather passes, design and validate numerical phase-B
projection with its production K-block/partial-rounding contract. Establish
the full 16-worker/L2 phase barrier before reusing any shared feature/weight
storage; the one-core alias sentinel does not close that requirement.
Audit final output joins, DMA retirement, and frame rearm together. Keep this
cut and existing transport proofs as regressions. Only a later matched
end-to-end measurement can establish a speedup; this draft makes none.

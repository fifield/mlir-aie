# One-core finite-phase alias sentinel validation

Tracked as `mlir-aie-2vb.2.3.4`, starting from baseline `4d405bf6a`.
Design contract: [PHASE_ALIAS_PLAN.md](PHASE_ALIAS_PLAN.md).

Status: the completion-fenced device build is frozen at
`/tmp/mdv6-phase-alias-tct.lLpDEe`. Compilation, independent resource/binary
gate, 11 host plus 5 IR CPU tests, exact 6/30/100/300-frame hardware gates,
and all seven configured lit tests pass. This proves the bounded one-core
finite-phase alias route, not numerical SPP or a full-worker phase barrier.
The earlier unsafe build `/tmp/mdv6-phase-alias.PcTuhi` was never executed on
hardware and is not accepted evidence. Do not use it as the final candidate.

## Bounded purpose

One core processes one phase-A-sized plane, exactly 25 phase-B-sized stripes,
then rearms for another frame. All phases overwrite the same explicit local
arena base. Opaque XOR tags expose ordering and stale-data errors without
introducing projection, pooling, rounding, or numerical-reference concerns.

This is distinct from the packet/gather milestones: those establish bounded
transport, whereas this cut tests finite task queues and explicit L1 overlap.
It does not yet compose those routes with this controller.

## Frozen host ABI and independent oracle

Artifacts are `phase_alias.xclbin` and `phase_alias.bin`; runtime arguments
are `[I, O]`, both native flat `uint16[217600]`, 435200 bytes apiece.

| Region | Element interval | Diagnostic output |
|---|---|---|
| A | `[0,12800)` | input XOR `0xA5A5` |
| B stripe `s`, `0 <= s < 25` | `[12800+8192*s,12800+8192*(s+1))` | input XOR `0x5A00 \| (s+1)` |

Each frame uses one persistent context, two persistent BOs, one native upload,
one runtime submission, and one output readback. No host phase/stripe loop,
intermediate synchronization, repacking, or metadata BO is part of the ABI.

The CPU oracle decodes each flat address into A or B and its stripe tag.
Independent slice-based CPU tests cross-check every word and all boundaries.
The runner compares every output uint16 exactly, including opaque NaN/Inf
encodings; there is no float conversion, tolerance, or checksum substitute.

Six cases cycle: low source-index half, high source-index half, random,
all-zero, all-one, and opaque edge bits. The two index halves jointly identify
all 217600 source words; neither half alone is unique. Random seeds and edge
rotations change across cycles to exercise rearm and reject stale output.

## Verified physical storage

The addressed map contains one `uint16[25120]` arena at core byte address
8192: `[8192,58432)`, 50240 allocated bytes. The 8192-byte stack occupies
`[0,8192)`. ELF text is 1728 bytes, with zero data and zero BSS bytes.

| Arena use | Physical core-local byte interval |
|---|---|
| A plane | `[8192,33792)` |
| B stripe | `[8192,24576)` |
| Reserved suffix beyond A | `[33792,58432)` |

The shared prefix is intentional overlap, not two logical buffers whose
placement happens to fit. The compiled-resource gate verifies one arena,
four distinct core BD IDs 0–3, six locks, and the stated stack/ELF sizes.
The suffix does not validate future weight or ping-pong subdivisions.

## Finite queue and six-lock ownership contract

Four terminating core tasks use RX S2MM1 and TX MM2S0. A RX/TX each transfer
12800 words once; B RX/TX each transfer 8192 words with repeat count 24,
meaning 25 executions. Queue A before B on each channel, two entries each.
Do not use a permanent B receive chain or an A+B chain repeated 25 times.
All four descriptors address arena offset zero and have distinct live IDs.

Six unique locks are required: `frame_empty` starts at 1; `Aready`, `Asend`,
`Bempty`, `Bready`, and `Bsend` start at 0.

1. A RX acquires `frame_empty`, writes A, and releases `Aready`.
2. Core acquires `Aready`, tags A, and releases `Asend`.
3. A TX acquires `Asend`, drains A, and releases `Bempty`.
4. Each B RX acquires `Bempty`, writes B, and releases `Bready`.
5. Core acquires `Bready`, tags that stripe, and releases `Bsend`.
6. Each B TX acquires `Bsend`, drains B, and releases `Bempty`.
7. After exactly 25 stripes, core consumes final `Bempty` and releases
   `frame_empty`, restoring the initial ownership state.

Only a completed A/B TX releases permission to overwrite its data. A 26th
queued B RX could steal the final token and prevent rearm; finite repetition
is therefore part of correctness, not merely a performance choice.

## Descriptor completion is separate from arena ownership

`dma_free_task` is allocator bookkeeping, not a device completion fence.
Arena locks establish safe data overwrite but do not alone prove a completed
DMA queue entry can be rewritten on the next host submission. Shim output
completion alone is not the intended proof of core task retirement.

The frozen implementation issues completion tokens on B RX and B TX only.
Core controller 27 has an explicit header-preserving TileControl0-to-shim
South0 packet route; the separate default shim controller remains 15.
Runtime awaits B RX channel 1 and B TX channel 0 device task completion, then
shim output completion. A-before-B queue order covers preceding A tasks.
Awaiting the B tasks also releases their compiler bookkeeping; explicit frees
then release input/A descriptors. Four core BDs, two BOs, and one submission
are preserved. `phase_alias_tasks.mlir` and `phase_alias_runtime.mlir` in the
build directory expose lowering for review. The completed gate verifies two
core completion tokens, one shim completion token, two queue entries per core
channel, and 25 B executions. It ties the checked runtime exactly to the
756-byte instruction binary containing 24 operations. This control-token and
binary check, followed by changing-frame hardware runs, establishes the
bounded implementation; compilation alone would not establish physical proof.

## Recorded gates and evidence

The 11 optimized host CPU tests cover exact ABI, independent oracle, paired
source identity, every phase/stripe boundary, stale phases, stripe reorder,
reused stripes, bypass, input preservation, persistent binding, runtime and
semantic failure poisoning, strict counts, and exclusive failure evidence.

Required per-frame counters are one run/returned/completed call, zero loads,
one 435200-byte upload, and one 435200-byte readback. Runtime/count/bit failures
stop immediately without retry and poison the object. When requested, NPZ
evidence stores input, expected output, available observed output, and error;
exclusive creation prevents silently overwriting an earlier failure.

Frozen SHA-256 identities, read directly from source/artifacts:

```text
aie2_phase_alias.py  3fc414af003b94ad31cda9a65d512c42e86c6f9245a658e741204bfa4cace863
phase_alias.cc      6b4d307dc0e43f832ae4fa0624054653f39d0fb9b5eb2e90b1cf5eb2f6c395b0
phase_alias.xclbin  36aa4da1b14c7eba8a6be6fa3a0a9b995f829253ede844a6cbb63c420243c6e5
phase_alias.bin     31fd5e48d186cfad9cfbe22c41fc85381a858309b3dc7c5bfb4fd98e7becd7da
```

`resource-gate.json` records PASS for the addressed map and exact binary tie.
All four serialized hardware runs passed every output word:

| Frames | Exact result | Warm mean wall, excluding frame 0 |
|---:|---|---:|
| 6 | PASS | 1.6260 ms |
| 30 | PASS | 1.8414 ms |
| 100 | PASS | 1.5793 ms |
| 300 | PASS | 1.6959 ms |

Evidence is `phase-alias-{6,30,100,300}.jsonl` in the frozen build directory.
These are diagnostic host wall times for copy/sync/run/readback, excluding
initialization, input construction, oracle, and validation. They are not a
matched baseline comparison or a full-model performance claim. Device cycles,
device DMA bytes, and driver context-switch counts were not measured.

Root's complete optimized CPU regression passed 184 tests. The local milestone
contributes 11 host tests and 5 compiled-IR tests. `lit-final.log` independently
records all seven configured gates PASS in 20.22 seconds: host checks, phase-A
shard, direct gather, packet aggregate, packet stripe join, packet gather, and
the new phase alias gate. The latter builds a fresh artifact for its run.

## Reproduce

From the MDV6 example directory, create a new isolated build:

```bash
source /home/jfifield/npu-dev-mdv6/env.sh
export PYTHONPATH=/home/jfifield/npu-dev-mdv6/install/mlir-aie/python:${PYTHONPATH:-}
phase_alias_build=$(mktemp -d /tmp/mdv6-phase-alias.XXXXXX)
make -C "$phase_alias_build" -f "$PWD/sppelan/Makefile.phase_alias"
python -O -m unittest sppelan.test_phase_alias_host sppelan.test_phase_alias_ir
python -O sppelan/check_phase_alias_ir.py --build-dir "$phase_alias_build"
```

Run the finalized compiled-resource checker and review completion-token
lowering before executing the following serialized, fail-stop hardware gates:

```bash
python -O sppelan/test_phase_alias.py --build-dir "$phase_alias_build" --frames 6 --failure-dir "$phase_alias_build/fail-6"
python -O sppelan/test_phase_alias.py --build-dir "$phase_alias_build" --frames 30 --failure-dir "$phase_alias_build/fail-30"
python -O sppelan/test_phase_alias.py --build-dir "$phase_alias_build" --frames 100 --failure-dir "$phase_alias_build/fail-100"
python -O sppelan/test_phase_alias.py --build-dir "$phase_alias_build" --frames 300 --failure-dir "$phase_alias_build/fail-300"
```

Retain JSONL stdout and stderr in the artifact directory. Stop on the first
failure; do not automatically continue after a timeout or reload the device.

## Remaining integration

This passing sentinel does not establish numerical SPP, the full 16-worker
phase barrier, L2 lifetime coordination, final phase-B output joins, or the
complete shared-weight arena. The next candidate cut is
[PHASE_A_GATHER_PLAN.md](PHASE_A_GATHER_PLAN.md), tracked as
`mlir-aie-2vb.2.4`: integrate all 16 numerical phase-A shards into resident
gather while preserving rounding metadata. That plan is not a frozen ABI or
validated implementation, does not close the full-worker barrier requirement,
and does not enable a production route or claim end-to-end SPP fusion.

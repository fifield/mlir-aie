# One-core finite-phase L1 alias sentinel

Tracked as `mlir-aie-2vb.2.3.4`, under phase/worker integration `mlir-aie-2vb.2.3`.

## Purpose and scope

The bounded proof is one phase-A-sized transfer followed by exactly 25
phase-B-sized transfers, then rearm for the next frame. Both phases use the
same physical core-local allocation. This isolates phase switching and DMA
ownership from arithmetic and the already separate packet/gather proofs.

Use new standalone generator, kernel, Makefile, host test, compiled-resource
checker, and configured lit test. Suggested artifact stem: `phase_alias`.
Do not modify the existing shard, gather, packet, or stripe-join proofs.

Update: this one-core proof now [passes exact 6/30/100/300-frame NPU gates](PHASE_ALIAS_VALIDATION.md).
The final implementation adds explicit B RX/TX task-completion tokens to the
original candidate below, fencing descriptor reuse independently of arena locks.
The next numerical integration is [phase A to resident gather](PHASE_A_GATHER_PLAN.md).
This does not validate a full sixteen-worker phase barrier or numerical phase B.

## Frozen candidate ABI and physical storage

Use two BO arguments, `[I, O]`, each native `uint16[217600]`:

- Elements `[0,12800)` are one full phase-A plane.
- The remaining 204800 elements are 25 phase-B stripes of 8192 elements.
- Stripe `s` starts at element `12800 + s * 8192`, for `s = 0..24`.

One input stream and one output stream carry that order. There is no metadata
BO initially; retaining four core descriptors is part of this bounded cut.
The input and output are each 435200 bytes per frame.

Place one `uint16[25120]` arena at core-local byte address 8192. Its range is
`[8192,58432)`, exactly 50240 bytes. Reserve stack `[0,8192)` and allocate no
second feature buffer. Inspect the ELF for additional data, scratch, or spills.

| Use | Arena-relative bytes | Physical core-local bytes |
|---|---|---|
| Phase A full plane | `[0,25600)` | `[8192,33792)` |
| Phase B in-place stripe | `[0,16384)` | `[8192,24576)` |
| Reserved arena suffix | `[25600,50240)` | `[33792,58432)` |

The suffix reserves space for a possible future combined arena. Its allocation
does not validate future weight, ping-pong, or final-output subdivisions.
Phase B intentionally echoes a 16-KiB stripe; it does not have the final
projection's output geometry.

## Diagnostic computation

Treat every lane as opaque uint16 bits, including NaN encodings. Phase A XORs
all 12800 lanes with `0xA5A5`. Phase B stripe `s` XORs all 8192 lanes with
`0x5A00 | (s + 1)`. These distinct fixed tags expose phase or stripe misordering.

The core has an outer frame loop: process A once, then an explicit bounded
`range(25)` for B, then rearm. Do not infer phase counts from data values.
The host independently computes the complete bitwise output and checks every
element. No float conversion, numerical tolerance, or compressed checksum.

## Four finite core descriptors

Use RX channel S2MM1 and TX channel MM2S0, with fixed routes throughout.
Reserve four distinct core BD IDs, for example:

| BD | Task | Length in uint16 | Repeat count |
|---|---|---:|---:|
| 0 | A RX | 12800 | 0 |
| 1 | A TX | 12800 | 0 |
| 2 | B RX | 8192 | 24 |
| 3 | B TX | 8192 | 24 |

Each task has one terminating BD: no self-loop or next-BD cycle. Repeat count
24 requests 25 executions of the B task, not 25 executions of an A+B chain.
Queue A before B on each channel; at most two task entries per channel are
needed. The descriptor address is the same arena base in all four cases.

Configure and enqueue these tasks in the device runtime instruction sequence
for each host submission. Configure one full-frame shim input and output task;
start their streams with all required core tasks available. In the implemented
proof, B RX and B TX each issue a task-completion token; await both, then the
shim output before returning. An explicit header-preserving core controller-27
route carries the tokens to the shim. There is no host synchronization between phases and
no second dispatch. Do not use an initialization-only queue as per-frame rearm.

## Ownership and frame rearm

Use six locks with unique core-local IDs: `frame_empty` initially 1, and
`Aready`, `Asend`, `Bempty`, `Bready`, `Bsend` initially 0. This is six locks
in total; all DMA descriptors have one acquire and one release.

1. A RX acquires `frame_empty`, writes the arena, and releases `Aready`.
2. Core acquires `Aready`, tags the A plane, and releases `Asend`.
3. A TX acquires `Asend`, reads the entire A plane, and releases `Bempty`.
4. Each B RX acquires `Bempty`, overwrites the same arena base, and releases
   `Bready`. Core acquires `Bready`, tags that stripe, and releases `Bsend`.
5. Each B TX acquires `Bsend`, drains that stripe, and releases `Bempty`.
6. After exactly 25 core iterations, core acquires the final `Bempty` and
   releases `frame_empty`, returning all other locks to their initial values.

Only A/B TX completion releases `Bempty`; only step 6 rearms `frame_empty`.
Thus the first B cannot overwrite A while A TX reads it, and the next A cannot
overwrite the last B. A finite B RX queue is essential: a 26th permanent receive
could steal the final `Bempty` token from the core and prevent rearm.

## API evidence and unresolved runtime questions

The repository exposes `repeat_count` in
[dma_start](../../../../python/dialects/aie.py:776). The
[AIERT queue translation](../../../../lib/Targets/AIERT.cpp:496) adds one to
the IR count. Its static DMA-start traversal queues tasks during configuration;
it is not an implicit nested per-frame phase controller.

The [runtime task definition](../../../../include/aie/Dialect/AIEX/IR/AIEX.td:1167)
accepts a tile, channel, direction, and repeat count. The compiler test
[dma_task_with_locks.mlir](../../../../test/bd-chains-and-dma-tasks/dma-tasks-to-npu/dma_task_with_locks.mlir)
checks lock-bearing core-tile tasks lowering to `npu.writebd`, as well as
memtile cases. This establishes local lowering support, not NPU2 execution.

The implementation's compiled and hardware gates resolve this one-core cut's
queue repetition and completion behavior. Its checker ties the reviewed
register-write sequence byte-for-byte to the actual instruction binary. Free
operations are compiler bookkeeping; B RX/TX device tokens are the reuse fence.
For any subsequent composition, recheck queue-depth and repetition, explicit
core BD assignment, channel enable/start lowering, and task lifetime handling.
Prove descriptor IDs cannot be reused or rewritten while an earlier task is
active. Inspect what task free/await operations lower to; do not assume that
shim-task lifetime rules automatically establish safe core-task reclamation.

## Gates and stop conditions

First compile only, in a fresh isolated directory. Inspect generated runtime
instructions and the addressed map: one core, one arena, own-L1-only allocation,
stack and ELF data non-overlap, exactly four core BDs, terminating chains,
A-before-B queues, correct repeat counts, and all six ownership locks.

Add CPU checks for exact ABI offsets, phase/stripe tag oracle, source mutation,
stale-frame rejection, and negative mutations of descriptor length, repeat
count, overlap address, premature release, and queue order. Run under `python -O`.

Freeze source and artifacts before serialized hardware tests. Start with at
least two changing frames to exercise A→25B→A; prefer six diverse patterns.
Then require exact changing-frame 30/100/300 runs and a fresh-build configured
lit gate. Stop on any timeout or mismatch and retain logs and failing inputs.

Do not claim SPP arithmetic, full 16-worker/L2 phase barriers, final phase-B
output joins, or the complete shared-weight arena from this sentinel. Those
remain follow-on integration gates even if this one-core proof succeeds.

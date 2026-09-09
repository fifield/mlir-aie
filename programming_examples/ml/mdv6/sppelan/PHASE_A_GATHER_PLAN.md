# Numerical phase A to resident gather

Tracked as `mlir-aie-2vb.2.4`. The separate integration task
`mlir-aie-2vb.2.3` remains incomplete until the full 16-worker phase barrier is
validated; the one-core alias sentinel does not close that requirement.

## Next bounded deliverable

Replace the diagnostic XOR producers in packet-gather with the validated
phase-A projection and three pooling stages. Deliver all 16 eight-channel
shards into the resident four-column gather in one host submission, preserving
rounding metadata. Suggested standalone artifact stem: `spp_phase_a_gather`.

This is a candidate design, **not a frozen ABI or validated routing schedule**.
Compile and inspect the finite queues before implementing the complete path.
Keep every existing proof as a regression. Do not modify production routing or
claim numerical phase B, end-to-end SPP fusion, or a performance gain here.

## Candidate host ABI

Four native-uint16 BO arguments, in order `[I, W, O, M]`:

| BO | Shape | Bytes |
|---|---|---:|
| I | `[400,256]` | 204800 |
| W | `[4 columns,4 workers,2 chunks,1056]` | 67584 |
| O | `[4 destinations,25 stripes,16 pixels,512 K]` | 1638400 |
| M | `[4 columns,4 workers,32]` | 1024 |

I has one logical host upload/sync, but four shim DMA tasks read the same BO:
actual input traffic into device memory is **819200 bytes**, not 204800.
Each column receives its own 16896-byte weight slice. Diagnostic output is
1638400 feature bytes plus 1024 metadata bytes. Do not describe replicated
diagnostic readback as the eventual fused model's required traffic.

Each shim has two input tasks on MM2S0, I then W, and two output tasks on
S2MM0, O then M. Input offsets for I are all zero; weight offsets vary by
column. Output and metadata offsets select that destination/column's slice.

## Worker arithmetic and physical memory

All workers read the same 400x256 activation and select distinct eight-output-
channel weight slices. Preserve two ordered K128 chunks, the intermediate
bf16 partial store, and BN/SiLU only after the last chunk. A physical chunk
contains 1040 logical words plus 16 padding words; its stride is 1056 words.
Do not replace this with full-K accumulation or change rounding mode.

Each worker receives weights once, then 25 activation stripes of 4096 words.
Project each stripe into the retained f0 plane. After all stripes, run three
local 5x5 max-pool layers with the existing finite-input/border contract.
The output allocation is one contiguous `uint16[12832]`:

- Words `[0,12800)` contain four 400x8 feature planes.
- Words `[12800,12832)` contain the existing rounding metadata.
- Pooling receives a metadata pointer into this suffix, not the grant buffer.

Send this allocation as one packet after receiving the worker's ordered grant.
There is no independent second full feature allocation.

| Per-worker allocation | Bytes |
|---|---:|
| Stack | 8192 |
| Features plus metadata | 25664 |
| One activation stripe | 8192 |
| Two padded weight chunks | 4224 |
| Grant token receive buffer | 64 |
| Total | 46336 |

Inspect actual addresses and ELF sections; the sum is not an allocation map.
This cut does not require phase-B L1 aliases. The one-core finite-alias proof
provides completion techniques, not evidence that these 16 workers alias safely.

## Memtile storage and channels

Each column retains one full I copy, its four weight sets, four worker return
packets, one gathered stripe, and one grant token:

| Per-memtile allocation | Bytes |
|---|---:|
| Input | 204800 |
| Weights | 16896 |
| Aggregate `[4,12832]` | 102656 |
| Gather stripe | 16384 |
| Grant token | 64 |
| Total | 340800 |

Keep S2MM0..3 for gather, S2MM4 for packet returns, S2MM5 for host ingress.
Use MM2S `col` for multicast gather source, MM2S4 for diagnostic O/M drain,
and MM2S5 for addressed weights, activations, and grants. This is six receive
and three transmit channels, not proof of physical switchbox routability.

## Finite distribution and grant schedule

Ingress has two finite descriptors: I acquires `frame_empty` and releases
`I_ready`; W acquires `I_ready` and releases `stage_ready`. Holding the frame
credit prevents either storage region being overwritten during downstream use.

MM2S5 queues three finite tasks, in this order:

1. Four addressed weight descriptors once, one per worker.
2. Four addressed activation descriptors, repeating the whole four-BD task
   25 times (`repeat_count=24`). All workers receive the same stripe per repeat.
3. Four grant descriptors once, with receive-completion ordering as in packet
   aggregation. A grant alone must never select a receive address implicitly.

Each activation descriptor transfers 4096 words and advances its own source
offset by 4096 after each invocation. Candidate dimensions are
`[(25,4096),(1,0),(1,0),(4096,1)]`, with a 4096-word BD length. This encoding is
**unproven** for the intended memtile task: inspect iteration size/stride and
enumerate all 25 addresses before hardware. Do not unroll 100 activation BDs.

The activation ring returns its first credit after every four sends. The first
grant may consume that credit only because its task is queued after all 25
activation repeats; the credit alone does not establish the phase boundary.

Core RX1 queues finite weights once, activation repeat24, grant once; core TX0
returns one packet. Require complete activation consumption and pooling before
the core signals send-ready. Preserve packet header removal and receive-gated
worker ordering. Audit all descriptor acquire/release pairs and queue capacities.

## Resident gather and metadata ownership

Four ordered S2MM4 returns fill aggregate rows of 12832 words. Gather reads
only feature words, using candidate source dimensions
`[(25,128),(4,12832),(4,3200),(128,1)]`, length 51200. The existing destination
scatter places each source's features in pixel/level/source/worker/lane K order.

Preserve the existing static gather source/scatter descriptors where possible.
Runtime DMA-task lowering treats the outer dimension as iteration separately;
static four-dimensional descriptors cannot be assumed to translate identically
to runtime tasks. If converting source gather, add an explicit compile gate for
transfer length, all 25 stripe addresses, iteration fields, and completion.

Source gather acquires `output_ready` and releases **`metadata_ready`**, not
`stage_empty` or `frame_empty`. Retain the aggregate until its metadata drains.

MM2S4 queues two finite tasks: stripe drain repeat24, then metadata drain once.
Each stripe acquires the last gather-turn lock and releases the first turn.
Metadata uses aggregate offset12800, length128, dimensions
`[(4,12832),(32,1)]`; it acquires `metadata_ready` and releases `frame_empty`.
The queued order guarantees all 25 local stripe drains precede metadata.

Approximate budget: 25 memtile BDs (2 ingress,12 sends,4 returns,1 source
gather,4 gather receives,2 drains), before any required adaptation. Check actual
allocation, initialization, queue depth, phase reset, and frame-to-frame reuse.
Static and runtime BD allocators are separate: reserve explicitly disjoint BD
IDs for both kinds on each memtile, and check the combined compiled map for
collisions rather than accepting either allocator's local success alone.

## Completion and numerical gates

### Compiler prerequisite discovered during implementation

`mlir-aie-96e` tracks a confirmed runtime queue-lowering defect: the generic
`aie-dma-to-npu` pass masks every start BD ID with `0xf`. NPU2 memtile
START_BD_ID is six bits, and odd channels require IDs 24–47. The compile-only
reproducer `repro_memtile_queue_id.mlir` emits value 8 instead of 24 at queue
register `0x1a065c` with the installed compiler. Do not execute that incomplete
reproducer on hardware.

The example-local implementation must emit explicit six-bit memtile queue
writes while retaining ordinary task descriptor lowering and completion waits.
Core/shim starts continue through their existing lowering. Check controller
IDs, physical queue addresses, start IDs, repeat counts and token bits against
the actual instruction binary. Do not remove the workaround until the shared
compiler is fixed, installed, and this entire numerical gate is rerun. This
work does not modify the shared compiler or installed toolchain.

### Required execution gates

Await all eight O/M shim tasks before freeing any input allocation. Add explicit
core/memtile task-completion tokens for finite queues before their BD IDs can be
reprogrammed. Device TCT waits remain inside the same instruction submission.
Compiler `dma_free_task` is allocator bookkeeping, not a device fence. Verify
the explicit control routes, token IDs, repeated-task completion semantics, and
actual lowered instructions; none is established by this proposal alone.

Build a fresh full-width production KB128 phase-A convolution as the numerical
reference. Apply three independently implemented CPU pooling levels and the
gather transpose to that result. Do not gate only against the reused shard
kernel. Check all 16 trained slices, finite structured/random inputs, negative
pool borders, signed-zero behavior, and exact metadata: M[...,0] is floor mode
0 and remaining words are zero. Report raw bf16 equality where expected and
the separately specified production/PyTorch tolerance, without conflating them.

Compile routing, resources, iteration addresses, and task lifetimes first;
then CPU oracle/mutation tests under `python -O`. Freeze artifacts before a
serialized small hardware smoke test, followed by exact changing-frame
30/100/300 campaigns and fresh-build configured lit. Stop and retain evidence
on timeout or mismatch. Final phase-B projection must later preserve its own
production K-block/partial-rounding contract; it is not implemented in this cut.

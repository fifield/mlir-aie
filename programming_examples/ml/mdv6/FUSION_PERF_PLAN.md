# MDV6 device-resident fusion implementation plan

Written 2026-09-09. Planning baseline: `ca0787a4d` on branch `mdv6`.
Implementation milestones below are proposed, not completed. Existing evidence
is recorded in [VALIDATION.md](VALIDATION.md); environment setup is in
[README.md](README.md). This document sets the forward performance direction.
[PERF_PLAN.md](PERF_PLAN.md) retains historical measurements and design notes.

## Objective and decision

Make a large improvement in full-model inference by retaining intermediate
data on device and greatly reducing host submissions, synchronization, and
context changes. Kernel fusion is the critical path. Shared regime artifacts,
runtime plumbing, and CPU-operator migration are enabling work whose benefit
may only appear after subsequent fusion and scheduling improvements.

A temporarily slower experimental implementation can be a successful milestone
if it proves a specific residency or dispatch capability and identifies the
next step that exploits it. Do not terminate the fusion effort because an
intermediate route fails the old per-category performance gate. Preserve the
working default while developing the new execution path explicitly.

The intended progression is:

```text
persistent executor + explicit tensor layouts
        -> whole-operator command sequences
        -> full-shape SPPELAN resident island
        -> repeated bottleneck / RepNCSP / RepNCSPELAN islands
        -> row-streamed larger shapes and adjacent blocks
        -> a few larger device execution regions
```

No particular FPS or 100–1000x speedup is established. Targets below concern
capabilities and graph structure; latency must be measured at each stage.

## What is already established

| Measurement, 2026-09-09 | Default | Restored R1–R3 |
|---|---:|---:|
| Warm wall, profile 7 | 1925.7 ms | 1967.9 ms |
| Host launches/frame | 453 | 466 |
| Selected xclbin paths | 32 | 16 |
| Runtime cached contexts | 32 | 19 |
| Cached instruction handles | 32 | 33 |
| Saved performance gate | PASS | FAIL: launch gap +43.3% |
| Changing-input validation | 30/100/300 frames PASS | 30 frames PASS |

The inventory counts are software observations, not independent hardware
residency measurements. R1–R3 reduced time inside the NPU runtime but increased
host overhead. It still runs the old host-driven dataflow, so these results do
not measure the final payoff of device-resident fusion.

Current code facts:

- [run_tiled_mc.py](run_tiled_mc.py) packs patches on the CPU, synchronizes
  input/weights, calls the synchronous runtime, then calls `out_buf.numpy()`
  and reconstructs a host output tensor. Both convolution and GEMM paths need
  a residency-aware replacement.
- [test_full_model_mc.py](test_full_model_mc.py), especially `run_rn_mc`,
  returns to CPU RepConv and residual addition inside every bottleneck.
- [kernels/rep_elan_bf16.cc](kernels/rep_elan_bf16.cc) consolidates kernel
  entry points into one object. This is useful linking infrastructure; it
  does not itself connect their activation dataflow.
- [conv/aie2_chain.py](conv/aie2_chain.py) connects two Conv1x1 workers through
  an ObjectFifo. Its test is only 8x8x16, uses the same weights for both stages,
  and is not a full-model fusion implementation.
- Workers already repeat until reconfiguration. Current weight acquisitions
  generally span a patch batch, not an entire model or guaranteed frame stream.
- The planner derives 125 logical NPU convolution invocations (65 3x3, 60
  GEMM) expanded into 453 launches. These exclude CPU RepConv/detection work.
  One submission per current convolution suggests an initial target near 125;
  one per major backbone/neck block suggests roughly 16 before cross-block
  fusion. These are structural targets, not implemented schedules. Moving CPU
  work onto the NPU can temporarily add kernels before they are fused.

## Three mechanisms to distinguish

| Mechanism | Removes | Still requires |
|---|---|---|
| Command batching within a coherent program | Repeated host submissions/waits | DMA and device stages may remain unchanged |
| Reuse of NPU-accessible external buffers | Host readback, copies and repacking when layouts agree | External-memory traffic remains |
| On-chip producer/consumer fusion | Intermediate external activation transfers | Explicit L1/L2 lifetimes, routing and synchronization |

On this installation, an already-NPU tensor's `.to("npu")` is a no-op, so its
buffer can be passed to a consumer without host access. Conversely `.numpy()`,
indexing, NumPy conversion, and even representation can cause readback. Verify
these semantics in the installed runtime when changing environments.

Use external buffers as deliberate spill storage where needed. On-chip chains
must execute within a context whose state is valid; do not assume L1/L2 data
or retained weights survive eviction/reconfiguration. A smaller context count
is a means of supporting execution regions, not the sole optimization target.

## Milestone 0 — persistent executor, layouts and measurement

Suggested new module: `mdv6_executor.py` (not present at planning time).
Separate construction from `run_frame(input)`:

1. Load the trained model and fuse/pack weights once.
2. Allocate named buffers and instruction streams once; bind contexts once.
3. Describe each tensor's logical shape, physical layout/strides, dtype,
   valid region, owner, producer, consumers, and last-use point.
4. Keep model parameters and buffer handles alive across frames. Define an
   explicit reload/reinitialization path after context loss or weight changes.
5. Make input upload and requested output download explicit frame boundaries.
   Internal execution must not silently convert device tensors to Torch/NumPy.

Prove a producer-output BO can feed a consumer with no host materialization.
If existing patch/OC-block layouts do not match, choose a producer DMA layout,
consumer access pattern, or device repacker. Do not conceal a host repack inside
the new executor API. Keep separate live buffers for concurrent operations;
the existing size-based pools assume synchronous execution and are not a
general liveness allocator.

Add an inference-only benchmark with model construction and CPU reference
outside the timer. Apply the same timing boundary to old and new executors.
Retain the old profile as a separate regression/history metric; never present
its approximately 1.93 s wall as directly comparable to a differently scoped
timer without measuring both paths under the new scope.

Collect submissions, host waits, context loads/evictions, sync calls and bytes,
device DMA bytes, layout-conversion bytes, and device cycles/trace where
supported. Distinguish cold configuration, frame execution, and validation.
`npu_run` is host-timed submit/wait work, not pure compute. `launch_gap` is a
residual bucket, not direct proof of the cost of individual runtime calls.

**Exit:** persistent frame API, documented buffer contract, and a two-stage BO
reuse test with zero intermediate host sync. Instrumented counts must agree
with the operations actually executed.

## Milestone 1 — whole-operator command sequences

Start with one existing 3x3 configuration, then one GEMM configuration.
Preserve useful tile geometry and OC/K blocking while moving the complete
spatial/OC loop into one generated runtime sequence. Supply a weight arena and
full-operator input/output buffers with explicit offsets. Drain only outputs
needed at the operator boundary.

Relevant generators are [conv/aie2_multicore.py](conv/aie2_multicore.py) and
[gemm_conv1x1/aie2_gemm_conv1x1.py](gemm_conv1x1/aie2_gemm_conv1x1.py).
Their build scripts should gain explicit experimental targets; do not replace
known-good artifacts or force the tiny shared R5 envelope onto the stem.

Generate a combined sequence from IR against one coherent placed program.
Do not concatenate existing `.bin` files: their program setup, addresses,
RTP writes, DMA mappings and completion assumptions can conflict. Bound the
number of outstanding descriptors and reuse them only after completion.
Internal fences may be necessary; the objective is to remove repeated host
round trips, not every device dependency.

Hold weights across longer tile loops when feasible. K-blocked GEMM currently
replays chunks for each patch; track the actual schedule and traffic rather
than assuming a host weight cache provides on-chip residency.

**Exit:** a full existing convolution executes through one host submission
(or an explicitly justified small bounded number), with matched numerical
results, counted transfers, and no CPU loop between its tile batches.

## Milestone 2 — full-shape SPPELAN resident island

First strengthen the small chain proof with distinct stage weights and multiple
tiles. Compare the same workload in three forms: ordinary host-materialized
execution, separate submissions sharing an external BO, and one connected
ObjectFifo program. This isolates what each mechanism removes.

Then implement the actual SPP9 shape:

```text
20x20x256 -> Conv1x1 -> f0:20x20x128
                       -> pool5x5 -> f1 -> pool5x5 -> f2 -> pool5x5 -> f3
     logical channels [f0,f1,f2,f3]:20x20x512 -> Conv1x1 -> 20x20x256
```

Input/output are 200 KiB each. Each retained feature is 100 KiB; all four are
400 KiB. A separate concat would add another 400 KiB. The two convolution
weight matrices total 320 KiB before BN data. Naively keeping input, output,
four features and concat requires 1200 KiB of activation storage, so enlarging
[sppelan/aie2.py](sppelan/aie2.py)'s single-core scratch buffers is unsuitable.

Use separate stages within one device program. A promising first schedule
shards pooling by channel: eight channels contain 6.25 KiB per complete 20x20
feature, or 25 KiB for four features. Pooling then needs no spatial halo exchange.
Convolution distribution and the final all-channel projection still need an
explicit gather/routing or partial-accumulation design; channel sharding alone
does not solve projection. Choose and document that design before compiling.

Treat concat as four logical channel groups consumed by the final K-blocked
projection, preserving accumulation order/rounding where required. Avoid a
second complete concat copy. Stream or distribute weights according to a
per-tile and per-memtile budget; aggregate array memory is not one shared pool.

If spatial tiling is selected instead, three pool stages produce halo radius
six: a 4x4 final patch can need a 16x16 f0 region. Pool padding must behave as
negative infinity, not zero, because SiLU outputs can be negative.

**Exit:** full-shape SPP9 uses one host submission, both convolutions and all
three pools run on device, and intermediate activations never return to host
or external memory. Only island ingress/egress activation transfers remain.
Eliminate the current logical 100 KiB f0 readback and 400 KiB concat upload;
measure padded DMA separately. Run changing-input and boundary tests.

SPPELAN is the manageable capability proof. Its current small share of frame
time is not the main eventual speedup; reusable machinery and removal of the
intermediate boundary are the first deliverables.

## Milestone 3 — bottleneck chain, RepNCSP, then RepNCSPELAN

CPU RepConv is a dependency barrier even if its CPU milliseconds are small.
Build at 20x20x64 in the following order:

1. One bottleneck: RepConv -> Conv+BN+SiLU -> plain residual add.
2. Three consecutive bottlenecks with intermediate activations retained.
3. Their enclosing RepNCSP, including entry/bypass/merge 1x1 convolutions.
4. A complete 20x20 RepNCSPELAN, including split branches and final merge.

Each 20x20x64 activation is 50 KiB. Residual/input, RepConv intermediate, and
Conv2 output total 150 KiB before weights. Each 64->64 3x3 matrix is 72 KiB.
Use channel blocking and tiled/streamed storage. For two successive stride-one
3x3 convolutions, a 4x4 output needs a 6x6 intermediate and 8x8 original region:
2, 4.5, and 8 KiB respectively at 64 bf16 channels. Retain the central original
input for the residual. At global boundaries, explicitly mask out-of-domain
intermediate values: BN/SiLU applied to padded input can produce nonzero values.

A single isolated bottleneck may still need the same 50 KiB ingress and egress
as today's Conv2 path. The three-repeat chain and absorption of adjacent 1x1s
are the planned steps that exploit residency. Judge this milestone sequence
together, while retaining measurements for each increment.

### RepConv numerical choices

The reference in [layers.py](../../../python/mdv6/layers.py) adds two linear
Conv+BN branches before one SiLU, with no identity branch. For eval mode and
the current bias-free convolutions, branch parameters can be folded in real
arithmetic:

```text
a = gamma / sqrt(running_var + eps)
b = beta - a * running_mean
W_fold = a3 * W3 + center_pad(a1 * W1)
b_fold = b3 + b1
output = SiLU(conv3x3(input, W_fold) + b_fold)
```

Validate this offline with trained weights before depending on it. bf16
quantization, intermediate rounding, accumulation order, and the current
approximate SiLU prevent assuming bitwise equivalence. Test cancellation-heavy
inputs and borders, then complete-model output. If folding exceeds the agreed
error budget, retain both branches inside the device island with no-activation
convolutions and a sum followed by one SiLU. Do not activate the branches
independently. Do not use `residual_add_silu_bf16` for the bottleneck's final
residual: that operation must be plain addition.

**Exit:** the repeated chain, then complete RepNCSP, has no internal host
dependency; an integrated RepNCSPELAN removes its previous CPU RepConv/add/
concat boundaries. Show correctness and counted transfer/dispatch reductions
at each expansion, followed by full-model measurements.

## Milestone 4 — larger shapes and adjacent execution regions

Apply the successful schedule to 40x40 and 80x80 using stripes/windows and
bounded live branch buffers. Full-tensor staging limits do not imply that all
larger intermediates must go external. Account for halo duplication, channel
gathers, padding, partial tiles, and the last consumer of every skip.

Then address the stem and ELAN. Conv0's padded input/output are each 6.25 MiB;
Conv1 output is 3.125 MiB. ELAN retains four 160x160x32 slices totaling 6.25 MiB.
These need streaming schedules rather than whole-map L1/L2 allocation. The
two stride-two stem convolutions have a 7x7 receptive field at stride four;
ELAN's two further 3x3s make the deepest path 23x23 in original-input space.
Its intermediate x3 is also a final-merge input and must remain available.

Use the real top-level dependencies:

```text
conv0 -> conv1 -> elan2 -> aconv3 -> re4(B3)
      -> aconv5 -> re6(B4) -> aconv7 -> re8 -> spp9(N3)
N4 = re12(concat(upsample(N3), B4))
P3 = re15(concat(upsample(N4), B3))
P4 = re18(concat(aconv16(P3), N4))
P5 = re21(concat(aconv19(P4), N3))
detect(P3, P4, P5)
```

The historical `aconv7 -> re8 -> spp9 -> re21` chain is not a valid consecutive
execution sequence: re21 needs the later head result. A valid first adjacent
region is `aconv7 -> re8 -> spp9`, exporting/retaining N3 for both consumers.
Move average pooling, nearest upsampling, residuals and layout-only operations
onto device as their boundaries enter an island. Detection can remain an
explicit terminal CPU boundary initially.

Choose a few execution regions and contexts from their dataflow and memory
requirements. Avoid fixing every stage to 32 cores: compare column allocations
and balance producer/consumer rates. Begin with a sequential schedule, then
overlap independent work only when buffer ownership and queue lifetimes are
proven. Optimize SiLU/vector loads and weight reuse using the resulting traces.

**Exit:** full-model measurements demonstrate benefit from the completed
resident path, then sustained operation under the final layout/context schedule.

## Planner work alongside the milestones

Extend [regime_planner.py](regime_planner.py) beyond grouped per-layer cost.
Its current repeated-layer aggregation and baseline-calibrated heuristic
cannot price fused islands correctly. Start with authored SPPELAN and
bottleneck-chain schedules, not a general compiler or placement search.

Required additions:

- Ordered operator instances, including current CPU work, and explicit tensor
  edges with layouts, dtype, valid regions and all consumers.
- Dispatch groups containing several device stages; one host submission is
  distinct from one kernel and from one external DMA transaction.
- Host/external-BO/memtile/core placement, live intervals, weight lifetimes,
  peak live memory and explicit spill/layout-conversion operations.
- Edge-based bytes, including halo duplication and transfers eliminated by
  fusion; do not keep charging the old host traffic on internal edges.
- FIFO rates, descriptor/barrier dependencies, finite buffering, context
  transitions, and the validity of retained state across those transitions.
- Numerical boundaries and folded/unfolded RepConv alternatives.
- A context budget and measured per-family/stage costs. Preserve uncertainty;
  do not extrapolate a promised FPS from array peak or the aggregate baseline.

Compare the same island as host-mediated, external-BO-resident, and on-chip
fused execution. Require the report to explain measured traffic/dispatch
changes before trusting its latency ranking.

## Validation and promotion policy

| Gate | Requirement |
|---|---|
| Semantic | Correct activation placement, tensor shapes, trained weights, finite outputs, boundary/padding behavior and accumulation rules |
| Resource | Compiled L1/code/L2 allocation and valid DMA/FIFO/barrier schedule; estimates alone are insufficient |
| Capability | Counted submissions, waits, contexts and bytes prove the intended boundary was removed |
| Performance | Same workload and timing scope; cold/warm, stage and full-model numbers reported separately |
| Sustained use | Changing inputs, repeated execution, retained-state validity and bounded-memory observations |

Set per-island tolerances against the existing computation before optimizing;
do not widen them merely to pass. The current full-model streaming gates are
max class/vector differences <0.5/<0.1 plus finite outputs. The older main
test's default <5.0/<5.0 gate alone is insufficient for this work. Keep numeric
metrics and add real-image detection comparisons before making quality claims.

Keep the default route and saved baseline intact. Experimental capability
milestones can pass despite slower latency when their resource/numerical gates
pass, their transfer/dispatch reduction is verified, and the next enabling
optimization is named. Performance categories may shift between host and
device; the old any-category >10% rule is diagnostic across architectures,
not the sole research acceptance criterion. Promote a completed path only
after matched end-to-end measurements show benefit and sustained correctness.

For a changed island: targeted edge/numerical tests, a short repeated hardware
run, then integration into the full model. For a stable integrated candidate:
30, 100, then 300 changing-input frames, advancing only after each passes.
Run hardware jobs serially and benchmark without compiler contention. Stop
on device error, preserve evidence and coordinate recovery; known timeout
corruption is not solved by ordinary successful runs. Do not deliberately
exercise the DMA-crash reproducer as part of performance benchmarking.

## Code map and reusable local examples

Paths below are relative to this directory unless stated otherwise.

| File | Use |
|---|---|
| `test_full_model_mc.py` | Authoritative current host dispatch and model comparison; `run_rn_mc`, `run_re_mc`, `main` |
| `run_tiled_mc.py` | Packing, buffer pools, routing, launch/readback, inventory; replace boundaries deliberately |
| `regime_config.py` | Existing contracts and explicit R1–R3 selector; reuse compatible envelopes |
| `conv/aie2_multicore.py`, `gemm_conv1x1/aie2_gemm_conv1x1.py` | Worker/RTP/DMA generation and whole-operator sequencing starting points |
| `kernels/rep_elan_bf16.cc` | Unified entry points and numerical/activation behavior |
| `conv/aie2_chain.py`, `conv/test_chain.py` | Small two-worker on-chip chain; needs distinct weights, multiple tiles and current build-path support |
| `sppelan/aie2.py`, `sppelan/sppelan_bf16.cc` | Small scalar reference structure; not a full-shape memory design |
| `bottleneck/aie2.py`, `repconv/repconv_bf16.cc` | Existing operator implementations; inspect scaling and numerics before reuse |
| `profile_harness.py`, `validate_stream.py`, `detection_validation.py` | Existing evidence and failure gates; preserve timing scope distinctions |
| `regime_planner.py` | Existing screening estimates and launch-count checks; extend with tensor graph/lifetimes |
| `../bottleneck/bottleneck.py` | Non-MDV6 on-chip 1x1/3x3/1x1 plus skip; row-window FIFO and weight-holding pattern, not a drop-in bf16 kernel |
| `../../basic/matrix_multiplication/single_core/single_core_iron.py` | Bounded DMA task-group overlap pattern |
| `../../basic/vector_scalar_add_runlist/` | Host runlist example; distinguish batching from on-chip fusion |

Runtime sources under the mlir-aie root: `python/iron/worker.py`,
`python/iron/runtime/runtime.py`, `python/utils/hostruntime/tensor_class.py`,
and `python/utils/hostruntime/xrtruntime/hostruntime.py`. Check the installed
copies actually imported by a hardware run, as source/install versions can
differ. Inspect placement reports instead of assuming the toy chain uses a
particular memtile route.

## Fresh-session starting procedure

1. Read this document, README and VALIDATION. Inspect `git status`, local
   instructions and relevant beads before changing files. Existing untracked
   weights, planner prompt and debug kernels are user work, not cleanup targets.
2. Source `/home/jfifield/npu-dev-mdv6/env.sh`, add installed mlir-aie Python
   to `PYTHONPATH`, and set `MDV6_BUILD_DIR` as shown in README. Confirm trained
   weights and the exact build root; missing source-tree xclbins are expected.
3. Select the baseline explicitly: `MDV6_REGIME_ROUTE=legacy`,
   `USE_REGIME_XCLBINS=0`, `USE_REGIME_KBLOCKED=0`. Reject diagnostic modes that
   return before full-model validation. If the environment/artifacts changed,
   establish a fresh short baseline before interpreting a regression.
4. Implement Milestone 0's buffer/executor contract and improve the existing
   two-stage chain proof. Agree on physical layouts before parallel kernel work.
5. Deliver the full-shape SPPELAN island and authored memory schedule. Next
   deliver the three-bottleneck chain; do not stop at standalone RepConv on NPU.

CPU checks currently available:

```bash
python3 -m unittest test_detection_validation test_validate_stream \
    test_regime_routes test_regime_planner
python3 regime_planner.py --cores 4 8 16 24 32 --output-dir /tmp/mdv6-planner
```

Existing baseline and streaming commands are in README. New executor/island
commands do not exist yet and must be documented as they are implemented.
Use unique evidence directories and separate experimental artifacts. Record
commit, dirty changes, imported runtime location, weights hash, build provenance,
flags, schedule, correctness metrics, dispatch/context counts, sync/DMA bytes,
memory observations and timing scope for every milestone. Temporary logs from
the earlier session may disappear; VALIDATION is the durable result summary.

Parallel implementation can assign one owner each to executor/measurement,
SPPELAN/device dataflow, and graph/lifetime modeling after agreeing on the
buffer contract. A single integrator reviews the interfaces and runs the NPU
gates serially. Independent review should check lifetime, halo, activation and
descriptor semantics before hardware execution.

## Tracking and session handoff

Relevant existing beads at planning time:

- `mlir-aie-mi7`: overall performance epic.
- `mlir-aie-2vb`, `mlir-aie-9oz`, `mlir-aie-k82`: on-chip pipelining, L2
  chains and spatial pipelines. Their old P3 ordering does not express this
  plan's critical-path priority; reconcile scope/priorities when claiming work.
- `mlir-aie-cup`, `mlir-aie-9xq`: RepConv and other CPU-operator migration,
  now scheduled as fusion prerequisites.
- `mlir-aie-1jg`: regime sharing, an enabler for the execution regions.
- `mlir-aie-976`: planner calibration and context budget.
- `mlir-aie-hll`: unnecessary context loads during variant probing; supporting
  executor cleanup, not a substitute for removing dataflow boundaries.
- `mlir-aie-mi7.2`: timeout recovery, still open.

Create/claim bounded implementation tasks under these items with the milestone
gates above; avoid duplicate broad epics. Update this document with completed
milestones, chosen layouts and measured results so a later session can resume
at the next unproven capability.

Follow the repository's landing workflow: commit scoped changes, pull/rebase
while preserving unrelated work, sync issues, push, and verify remote parity.
The installed beads version uses `bd dolt push`; `bd sync` is unavailable.
Do not erase user changes or stashes as incidental cleanup. A handoff must say
what capability is proven, what remains unproven, and the next concrete test.

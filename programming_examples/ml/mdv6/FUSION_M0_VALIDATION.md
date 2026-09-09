# Fusion execution foundation — 2026-09-09

This increment implements the persistent **host-materialized** baseline and a
small three-route residency proof. It does not complete full-model residency
or the full Milestone 0 buffer-prebinding contract. The roadmap remains
[FUSION_PERF_PLAN.md](FUSION_PERF_PLAN.md).

## Implemented interfaces

- `test_full_model_mc.py`: extracted `load_model`, `pad_conv0_weights`,
  `run_hybrid_forward`, and `report_comparison`; the original `main` and
  profiler still use the same graph and comparison.
- `mdv6_executor.py`: `MDV6Executor().run_frame(x)` retains the trained model,
  fused weights and padded stem weights. Input is contiguous CPU bf16 NCHW
  `(1,3,640,640)`; outputs are host detection triples. Parameters are immutable.
  Lazy global runtime caches/pools remain synchronous and shared process state;
  this is not a concurrent executor. After a failed frame it rejects reuse.
  Recover the device as appropriate and restart the process after context loss
  or weight changes; it does not implement safe in-process context recovery.
- `benchmark_executor.py`: persistent and legacy-reconstructed routes use
  the same forward timer. Construction, CPU reference, comparison, snapshots
  and reporting are outside; CPU islands and layout work remain inside.
  The cold frame includes lazy context/buffer initialization. Legacy mode
  reconstructs before each reference and leaves stem packing inside forward.
  Run each route in a separate process. Both include counter-hook overhead.
- `device_buffers.py`: contiguous bf16-as-uint16 external BO contract with
  logical/physical shape, strides, valid region, owner, producer/consumers and
  last-use metadata. A guarded buffer requires matching layouts and explicit
  valid contents before handoff. It rejects implicit NumPy conversion and
  mismatched dtype/size. Last-use metadata is not an asynchronous allocator.
- `runtime_metrics.py`: scoped, nonnested, single-threaded API observation.
  Counts attempted/returned/successful runtime calls, loads, cache misses,
  software eviction calls, and tensor sync calls/whole-BO bytes. It restores
  methods after exceptions. Direct pyxrt syncs outside the tensor API, driver
  context transitions, hardware DMA and cycles are not observed. Unavailable
  fields are null; cache misses are not claimed as hardware context switches.

## Full-model evidence

Trained weights and prebuilt baseline artifacts match [VALIDATION.md](VALIDATION.md).
Weights SHA-256:
`4d0af6e3d80bbbdbcc22a5ec6cf997f66e3f6062e463a04c1da34cf08f44c3ae`.
Installed runtime:
`/home/jfifield/npu-dev-mdv6/install/mlir-aie/python/aie/utils/hostruntime/xrtruntime/hostruntime.py`.
Default route explicitly selected; no GEMM override. This increment was tested
as working-tree changes atop `ab4582988`; the commit adding this document
contains the tested implementation. Baseline xclbins were not rebuilt.

| Inference-only check | Result |
|---|---|
| Legacy reconstructed, 7 frames, seeds 42–48 | PASS; 1597.3 ms mean of 6 warm frames |
| Persistent, 30 frames, seeds 42–71 | PASS; 1583.2 ms mean of 29 warm frames |
| Persistent first 6 warm frames, matched seeds | 1585.7 ms |
| Persistent maximum class/vector difference | 0.238037109375 / 0.03125 |
| Per-frame completed submissions, both routes | 453 |

This small timing difference is not a demonstrated major speedup. Do not
compare these approximately 1.59 s measurements directly with the historical
approximately 1.93 s profile wall: their scopes differ. The routes used separate
processes, without compiler/NPU-job contention, but were not randomized trials.

Warm-frame API counts agree across both routes: 810 tensor uploads totaling
424,077,184 bytes and 453 downloads totaling 87,130,112 bytes. These include
padding/replayed weights and are **not** logical graph-edge or hardware DMA
counts. Twenty runtime load calls per warm frame hit existing contexts: zero
context-cache misses and zero software eviction calls. Thus persistent model
ownership alone has not removed internal transfers, repeated runtime loads,
or submissions. The next work must change the device dataflow/schedule.

The original `test_full_model_mc.py --profile 7 --baseline
profile_baseline.json` also passes all seven numerical checks and the saved
category gate: 1913.3 ms warm wall, 453 launches, launch gap +8.3% versus saved
baseline. An earlier three-frame run failed the launch-gap category at +11.0%;
CPU checks overlapped that short profile. The seven-frame follow-up was isolated.
Both results are retained in `/tmp/mdv6-fusion-legacy-profile{,-7}.{log,json}`;
the short-run failure is not evidence of a numerical failure.

The persistent 100/300-frame stages, RSS/liveness tracing and real-image
quality comparisons have not run. Earlier 300-frame legacy evidence does not
prove this executor's indefinite memory behavior. No timeout/recovery test was
attempted, and no default route or saved baseline was replaced.

## Distinct-weight, multi-tile residency proof

`conv/aie2_bo_reuse.py` builds single-stage and connected two-stage Conv1x1
+BN+approximate-SiLU programs. Both stages use distinct weights. The physical
activation contract is three independent `8x8x16` tiles, contiguous
`[tile,H,W,C]`, bf16 bits in uint16: 3072 elements / 6144 bytes. Weight packing
is `[O,I]` followed by 16 BN scales and 16 biases. Each worker holds weights
across three tiles per submission, not across frames.

Two artifacts were freshly compiled using `conv/Makefile.bo_reuse` into
`/tmp/mdv6-bo-reuse.qzav2H`. Ten changing inputs (seeds 1042–1051) passed,
including a final run under `python3 -O`. All three routes agree bitwise;
maximum error against the approximate-SiLU CPU oracle is 0.0078125, below
the predetermined 0.05 gate. Inputs change and outputs are checked to change.

| Route | Submissions | Intermediate upload | Intermediate download |
|---|---:|---:|---:|
| Host-mediated | 2 | 6144 B | 6144 B |
| Same-context external BO handoff | 2 | 0 | 0 |
| Connected ObjectFifo | 1 | 0 | 0 |

Sync counts come from instrumenting actual XRT tensor methods between ingress
and final output download. Runtime call and successful-return counts are both
checked. The external route reuses the identical intermediate tensor, without
host repacking; it does not prove sharing between distinct contexts.

Compiled `chain.mlir.prj/input_with_addresses.mlir` places workers on `(0,2)`
and `(0,3)`. Intermediate `activation1` uses two 2048-byte buffers on `(0,2)`,
accessed through neighboring shared L1. This is **not a memtile chain**.
The runtime sequence has four shim DMA descriptors: input, output, and two
weights. No intermediate external DMA is generated. Eliminating the logical
6144-byte external write plus read follows from that compiled schedule, not
from a hardware bandwidth trace. All allocations/placement compiled successfully.

The initial fixed-order experiment averaged 4.287 / 4.158 / 2.896 ms over nine
warm frames for host / external / connected routes. This timer excludes ingress
and final download and alternates contexts in a fixed order. These are small
capability-test observations, not full-model latency predictions.

## Reproduce and extend

Use the complete environment setup in [README.md](README.md), then:

```bash
python3 benchmark_executor.py --route legacy --frames 7 \
    --report /tmp/mdv6-legacy-new.jsonl
python3 benchmark_executor.py --route persistent --frames 30 \
    --report /tmp/mdv6-persistent-new.jsonl

# A new directory isolates experimental artifacts from the production build.
proof_build=$(mktemp -d /tmp/mdv6-bo-proof.XXXXXX)
make -C "$proof_build" -f "$PWD/conv/Makefile.bo_reuse"
python3 -O conv/test_bo_reuse.py --build-dir "$proof_build" --frames 10

python3 -m unittest test_detection_validation test_validate_stream \
    test_regime_routes test_regime_planner test_mdv6_executor \
    test_runtime_metrics test_device_buffers
```

Report paths are exclusive; choose new ones. JSONL records include timing scope,
seed, thresholds, weights hash, build root, flags, installed runtime and counters.
Record source commit/dirty state and artifact provenance with the report.
Original evidence is `/tmp/mdv6-executor-{legacy-7,persistent-30}.{log,jsonl}`
and `/tmp/mdv6-bo-reuse.qzav2H/proof{,-final}.jsonl`; these are temporary.

The CPU suite has 53 passing tests and configured `run_host_checks.lit` passes.
Configured `run_fusion_primitives.lit` also passes, building and exercising the
isolated proof (log `/tmp/mdv6-fusion-primitives-lit.log`); full-model
multi-frame lit retains the previous regression workflow.

Next tasks: `mlir-aie-2vb.2` authors the SPP9 buffer/live-memory and projection
schedule; `mlir-aie-2vb.3` moves one existing 3x3 operator's complete tile loop
into a coherent bounded device command sequence, then extends it to GEMM.
Keep the working tiling and default route. The small chain is reusable evidence,
not justification to bypass full-shape routing, halo and resource gates.

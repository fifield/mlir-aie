# MDV6 validation — 2026-09-09

The default hybrid CPU/NPU route passes the performance gate and 30/100/300
changing-input frames. The restored R1–R3 route passes numerical checks and
reduces cached contexts, but fails the saved performance gate. Keep it opt-in.

## Provenance

- Integrated `origin/mdv6` through `3c12544c8` (multi-frame lit coverage).
- Timing precision committed as `ab6f95ab3`; reviewed implementation and host
  tests committed as `4054e7192`. The 300-frame process began with that source
  content before its commit was created. Earlier 30/100-frame runs used the
  initial validator, with identical seed progression and numerical thresholds;
  the final validator adds richer metrics and failure-path checks.
- Strix Halo NPU2; Linux `6.17.0-20-generic`; environment and commands are in
  [README.md](README.md).
- Build root:
  `/home/jfifield/npu-dev-mdv6/build/mlir-aie/programming_examples/ml/mdv6/test_stx`.
  Baseline artifacts were prebuilt. All nine selected R1–R3 envelopes were
  built in this session: three convolution and six GEMM builds, zero failures.
  This does not claim a fresh rebuild of every baseline/layer artifact.
- Trained weights SHA-256:
  `4d0af6e3d80bbbdbcc22a5ec6cf997f66e3f6062e463a04c1da34cf08f44c3ae`.

## Isolated performance comparison

Both final runs used `test_full_model_mc.py --profile 7 --baseline
profile_baseline.json --save-baseline <separate-output>`: one cold frame and
six measured warm frames, identical seed 42 and weights. All seven numerical
checks passed in each run. No compiler builds or other MDV6 runs overlapped
these profiles. The committed baseline was not replaced.

| Metric | Saved baseline | Default now | R1–R3 now |
|---|---:|---:|---:|
| Warm wall, ms | 1869.0 | 1925.7 | 1967.9 |
| Launches/frame | 453 | 453 | 466 |
| Runtime run bucket, ms | 1086.1 | 1102.4 | 1005.7 |
| Launch gap, ms | 404.4 | 437.7 | 579.3 |
| NumPy assembly, ms | 34.1 | 34.8 | 36.5 |
| CPU layers, ms | 51.3 | 52.3 | 53.8 |
| Pre/post, ms | 293.1 | 298.6 | 292.7 |
| Saved category regression gate | — | PASS | FAIL |

R1–R3 is 2.2% slower than the fresh default. Its launch-gap bucket regresses
43.3% against the saved baseline, exceeding the 10% category gate despite
lower time inside `DefaultNPURuntime.run`. That run bucket includes runtime
submit/wait work; it is not a hardware-only compute measurement.

The initial three-frame baseline at `ab6f95ab3` also passed: 1916.4 ms warm,
453 launches, class/vector differences 0.2260/0.0312.

## Changing-input validation

Each process reconstructs the model, loads trained weights, increments the
input seed from 42, computes a CPU reference, and compares the hybrid result.
The gates are finite outputs on every detection scale and maximum absolute
class/vector differences below 0.5/0.1. No timeout, EIO, or driver reload
occurred in these runs.

| Route | Frames | Seeds | Result | Launches/frame |
|---|---:|---|---|---:|
| Default | 30 | 42–71 | PASS | 453 |
| Default | 100 | 42–141 | PASS | 453 |
| Default | 300 | 42–341 | PASS | 453 |
| R1–R3 | 30 | 42–71 | PASS | 466 |

The 300-frame default run's maximum class/vector differences were
0.2412109375/0.03125. R1–R3's were 0.236083984375/0.03125.

| Software inventory per frame | Default, 300 frames | R1–R3, 30 frames |
|---|---:|---:|
| Selected xclbin paths | 32 | 16 |
| Cached instruction handles | 32 | 33 |
| Runtime cached contexts | 32 | 19 |

These are software-cache observations, not independent driver-residency
measurements. The installed runtime caches contexts by xclbin path and mtime.
`_get_mc_variant` calls `_load_handle` to probe availability before regime
dispatch, which can retain unused standalone contexts. Thus selected artifacts
and cached contexts are different quantities. See `mlir-aie-hll`; the probes
have not been established as the cause of all extra launch-gap time.

Default RSS was 532.4 MiB after frame 1, 596.7 at frame 30, 605.2 at frame
100, 606.7 at frame 200, and 612.8 at frame 300. This shows substantial warmup
allocation and modest later growth, not proof of leak-free indefinite use.
Mean full-validation wall time for frames 2–31 was 1815.1 ms and for the final
30 frames was 1821.0 ms. This timer includes model setup and CPU reference
and excludes explicit garbage collection; it is not the profile wall metric.
Some of the 100-frame run and the start of the 300-frame run overlapped CPU
compilation, so those runs are stability evidence, not isolated benchmarks.

R1–R3 has not passed a 100/300-frame gate. No deliberate timeout/DMA corruption
test or real-image detection-accuracy evaluation was performed. The longer
default run does not close the known timeout-recovery issue.

## Planner and host checks

All 31 CPU tests passed, including 11 planner tests. The configured build-tree
`run_host_checks.lit` passed using `lit -v`. Numerical tests cover nonfinite
reference/output tensors, later scales, anchors, and shape/count mismatches.
Stream tests cover early diagnostic exits, failed frames, exceptions, flushing,
runtime restoration, report preservation, and seed progression.

The offline planner reproduces 453/742/757/933 launches for the recorded
baseline/R5/shared-K/shared-convolution configurations without fitting those
regression counts. Its R1–R3 prediction of 466 launches is confirmed by this
session's hardware runs. The old summary of 453 for that subset is not
applicable to these contracts. The difference is +15 non-K GEMM calls and
two K-blocked call savings.

The planner is a screening tool. Its aggregate-baseline latency model predicts
the large regression directions but underestimates their magnitude and did
not predict the small R1–R3 wall regression. Memory-fit estimates require
compiler validation. Per-family calibration and a context-constrained
selection model are tracked in `mlir-aie-976`.

## Local evidence and handoff

Temporary session evidence remains at:

- `/tmp/mdv6-review-baseline.{log,json}`
- `/tmp/mdv6-final-default.{log,json}` and `/tmp/mdv6-final-r1-r3.{log,json}`
- `/tmp/mdv6-stream-{30,100,300}.{log,jsonl}`
- `/tmp/mdv6-r1-r3-stream-30.{log,jsonl}`
- `/tmp/mdv6-regime-{conv,gemm}-build.log`
- `/tmp/mdv6-planner/` (JSON/CSV screening reports)

These files are temporary local evidence; the findings above are the durable
summary. The full rebuilt multi-frame lit hardware test was not rerun; its
profile and changing-input commands were exercised directly against the
specified artifacts. Keep the default route. Next work is `mlir-aie-hll`
(context probes) and `mlir-aie-976` (planner calibration), followed by another
isolated comparison before considering deployment of shared regimes.

# MDV6 on Strix Halo NPU

This example implements the MDV6-mit-yolov9-c full-model forward pass as a
hybrid CPU/NPU pipeline using bf16. Fused convolution, batch normalization,
and SiLU use up to 32 AIE2P compute tiles. RepConv, pooling, upsampling, and
detection retain CPU work. It is a numerical comparison and profiling example;
it does not evaluate detection quality on a labeled image dataset.

The forward performance roadmap is [FUSION_PERF_PLAN.md](FUSION_PERF_PLAN.md):
persistent execution, fewer dispatches, and device-resident fused islands,
with implementation milestones and fresh-session instructions.

The first implemented foundation and reproducible commands are in
[FUSION_M0_VALIDATION.md](FUSION_M0_VALIDATION.md): persistent hybrid execution,
matched inference-only timing, and a three-route small-chain residency proof.
Full-model device-resident fusion is still in progress; the default is unchanged.

The next implemented step is [whole-convolution sequencing](FUSION_M1_VALIDATION.md):
an opt-in route removes 36 full-frame submissions by batching output-channel
blocks. The document records its build/run commands and validation limits.

## Run with existing artifacts

These paths describe this checkout's development environment. Adjust them
together for another installation. Prerequisites are the Strix Halo NPU2 device
and working XRT driver, installed mlir-aie Python package, PyTorch, trained
weights, and compiled multicore/GEMM artifacts.

```bash
source /home/jfifield/npu-dev-mdv6/env.sh
export PYTHONPATH="/home/jfifield/npu-dev-mdv6/install/mlir-aie/python${PYTHONPATH:+:$PYTHONPATH}"
export MDV6_BUILD_DIR=/home/jfifield/npu-dev-mdv6/build/mlir-aie/programming_examples/ml/mdv6/test_stx
cd /home/jfifield/npu-dev-mdv6/mlir-aie/programming_examples/ml/mdv6

# Select the default per-shape route explicitly.
export MDV6_REGIME_ROUTE=legacy
export USE_REGIME_XCLBINS=0
export USE_REGIME_KBLOCKED=0

# Compare one full-model forward pass with the PyTorch reference.
python3 test_full_model_mc.py

# One cold warmup, two measured warm frames, and the saved regression gate.
python3 test_full_model_mc.py --profile 3 --baseline profile_baseline.json \
    --save-baseline /tmp/mdv6-profile.json
```

Stage `mdv6_bf16_weights.pt` alongside `test_full_model_mc.py`. It must contain
the compatible trained state dictionary. The script also has a machine-specific
fallback to `/home/jfifield/mdv6/mdv6.pt` (TorchScript), but lit requires the
adjacent state-dictionary file. Random initialization is not a useful baseline
and has historically produced NaNs. Confirm that the test prints that trained
weights were loaded.

Artifacts live under `$MDV6_BUILD_DIR/mc` and `$MDV6_BUILD_DIR/gemm`, not in the
source directory. A source tree without xclbins is expected. `env.sh` alone
does not put the installed mlir-aie Python package on `PYTHONPATH`.

Keep `profile_baseline.json` unchanged when collecting results: use a separate
output path as above. Record `git rev-parse HEAD`, local changes, environment
flags, artifact location/build provenance, and complete test logs with results.
Reusing prebuilt artifacts validates those artifacts; it does not prove a fresh
build of the checked-out generator sources.

## Build and regression integration

With the environment above and the Peano compiler available, build artifacts
using:

```bash
python3 conv/build_multicore.py
python3 gemm_conv1x1/build_gemm_conv1x1.py
```

Inspect build failures. The lit wrapper currently tolerates the multicore
builder's nonzero exit because an unused configuration has a documented L2
overflow; a missing configuration actually selected by the full model is a
failure. Do not infer a clean build from the wrapper continuing.

`run_full_model.lit` builds into its own `test_stx` directory and runs one
forward pass. `run_full_model_multi_frame.lit` builds into a separate
`test_stx_multi_frame` directory and runs `--profile 3` followed by
`validate_stream.py --frames 3`, exercising repeated model creation, cache reuse,
and changing inputs. Both require `ryzen_ai_npu2`, `peano`, `torch`,
and `mdv6_weights`. Per-layer tests also live in the layer subdirectories.
The multi-frame lit test does not pass `--baseline`; numerical regression and
performance regression are separate checks.

## Evidence and limits

The committed `profile_baseline.json` records 1869.013 ms warm-frame wall time
and 453 launches (about 0.54 fps). `PERF_PLAN.md` records numerical differences
around 0.226 for classes and 0.031 for vectors. These are historical results.

See [VALIDATION.md](VALIDATION.md) for the current session's hardware results,
artifact provenance, streaming stages, and measurement caveats.

The profile gate rejects more than 10% regression in measured nonzero baseline
categories. Three frames provide only two warm samples, so host timing noise
can affect that gate. The original full-model comparison uses fixed random
input (seed 42), trained weights, and maximum class/vector differences below
5.0; it is not a detection-accuracy test. Repeated identical input does not
cover stale-output bugs as strongly as changing-input validation.

For changing-input streaming validation, use the same environment and run
the following sequentially, advancing only after each run passes:

```bash
python3 validate_stream.py --frames 30 --report /tmp/mdv6-stream-30.jsonl
python3 validate_stream.py --frames 100 --report /tmp/mdv6-stream-100.jsonl
python3 validate_stream.py --frames 300 --report /tmp/mdv6-stream-300.jsonl
```

Choose unused report paths: the validator creates them exclusively to preserve
earlier evidence. It advances the seed each frame, checks finite detection
outputs, uses stricter maximum class/vector differences of 0.5/0.1, and records
per-frame wall time, RSS, launches, and cached runtime handles. Cached artifact
names are an observation of software state, not an independent measurement of
live device contexts. The validator reconstructs the model and runs its CPU
reference each frame; its wall time includes that work and is not directly
comparable with the profile harness's warm-frame wall time. Use the profile
harness for latency comparisons and avoid concurrent compiler builds.

Known kernel timeouts can leave the NPU requiring a driver reload. Stop hardware testing
on a timeout/device error, preserve diagnostics, and coordinate recovery.

## Experimental R1–R3 route

Regime xclbins are experimental and default off. Historical R1–R3 sharing
preserved performance, but R5 retile and cross-regime shared envelopes raised
launch count and latency substantially. The explicit subset selector enables
R1–R3 convolution and GEMM members, including K-blocked GEMM, while excluding
R5 and cross-regime shared envelopes. It overrides the old two flags, so set
`MDV6_REGIME_ROUTE=legacy` and both flags to zero to restore the default path.

Build the subset artifacts in the configured build root before selecting it:

```bash
python3 conv/build_multicore.py \
    regime_r1_conv3x3 regime_r2_conv3x3 regime_r3_conv3x3
python3 gemm_conv1x1/build_gemm_conv1x1.py \
    regime_r1_gemm_non_k regime_r1_gemm_kblocked \
    regime_r2_gemm_non_k regime_r2_gemm_kblocked \
    regime_r3_gemm_non_k regime_r3_gemm_kblocked
MDV6_REGIME_ROUTE=per-regime-r1-r3 python3 test_full_model_mc.py --profile 3 \
    --baseline profile_baseline.json --save-baseline /tmp/mdv6-r1-r3-profile.json
```

The default artifacts are still needed for layers outside the subset. The R1
GEMM builder also emits instruction variants that this route does not use.
Consult `VALIDATION.md` before treating restored routing as equivalent to the
historical performance result.

## Offline planner

```bash
python3 regime_planner.py --cores 4 8 16 24 32 \
    --output-dir /tmp/mdv6-planner-review
```

This runs without Torch, XRT, or hardware and writes JSON/CSV reports. It
screens existing envelopes for padding, launch count, transfers, occupancy,
and estimated memory fit. Calibration uses only `profile_baseline.json`;
latency estimates are heuristics, and memory estimates are not compiler proof.
The smaller-core candidates assume hypothetical rebuilds with unchanged PPC.

At 32 cores, the model predicts 453 launches for the standalone baseline, 742
after R5, 757 after shared K-blocked GEMM, and 933 for all shared envelopes.
The restored R1–R3 subset predicts 466 launches, so the old 453-launch R1–R3
measurement cannot be assumed to apply to the current code. Hardware evidence
and any discrepancy are recorded in `VALIDATION.md`. Read the report's
`limitations` before using scores to choose new configurations.

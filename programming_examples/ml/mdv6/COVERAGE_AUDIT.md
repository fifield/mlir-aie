# MDV6 implementation and test coverage audit

Audited 2026-09-10 against `f7cb4c366`, including the local working tree.
This is a source/registration audit plus fresh hardware-free test execution;
hardware results cited in validation documents were not rerun.

**MDV6 has substantial lit coverage, but not comprehensive layer/variant
coverage.** Its strongest tests protect the hybrid forward, selected whole
operators, and recent routing/lifetime primitives. Small standalone layer tests,
unregistered manual variants, shared numerical oracles, and incomplete resident
fusion leave significant gaps. No meaningful line/branch coverage percentage
for the Python runtime or AIE kernels is currently available.

## Inventory and fresh verification

Tracked files under this example at the audited revision:

| Surface | Count | Meaning |
| --- | ---: | --- |
| Python files | 125 / 21,591 lines | Includes implementations, generators, builders, tests and checkers |
| `test*.py` files | 63 / 10,244 lines | Includes manual hardware scripts and reference-only smoke tests |
| `aie2*.py` generators | 31 | Multiple historical and experimental implementations |
| C++ sources | 22 / 4,682 lines | Scalar, vector, consolidated and diagnostic kernels |
| C++ headers | 1 / 16 lines | Constant-index BF16 row extraction helper |
| lit entries | 25 | Confirmed by configured `lit --show-tests` |

Counts exclude the shared `python/mdv6` reference package and untracked files.
The working tree also contains untracked weights, `test_xrt_context_switch.py`,
`gemm_conv1x1_4x8x8.cc`, and `passthrough_test.cc`; these are not committed
regression coverage. Test-file/source-line ratios are not execution coverage.

Fresh execution in this checkout's configured environment:

- `run_host_checks.lit`: **205 unittest methods in 27 modules passed**, plus
  the route reducer's four parser/identity/selection/guard checks under `python -O`.
- `run_gemm_completion.lit`: **7 unittest methods passed**, including generated
  IR completion ordering and build freshness.
- Both actual lit wrappers passed together: **2 selected / 25 discovered**.
  The other 23 were excluded by the audit's filter, not run or claimed passing.
- No NPU numerical tests, full artifact rebuild, or detection dataset evaluation
  was performed in this audit.

The test mix is nine basic layer hardware entries, two full-model hardware
entries, one host suite, one generated-IR suite, one fusion-primitive hardware
entry, three whole-operator hardware entries, one sparse arithmetic hardware
probe, and seven SPP prototype hardware entries. One of the latter seven is
intentionally feature-disabled pending routing feasibility.

## What the full model actually implements

The CPU reference is `../../../python/mdv6/model.py` and `layers.py`. The active
hybrid graph is `test_full_model_mc.py:368`, dispatched through
`run_tiled_mc.py`. `mdv6_executor.py` wraps that graph in a persistent model
lifetime; its module documentation explicitly says every convolution still
materializes its output on the host.

| Graph blocks | Execution in the default hybrid route |
| --- | --- |
| Stem Conv0/1 | NPU fused convolution/BN/SiLU; input channels padded 3 to 8 |
| ELAN2 | Four NPU convolution blocks; host split/concat |
| AConv3/5/7/16/19 | CPU average pool followed by NPU fused convolution |
| RepNCSPELAN4/6/8/12/15/18/21 | NPU projections and bottleneck second convolutions; CPU RepConv, residuals, split/concat |
| SPP9 | Two NPU GEMM projections; CPU three max pools and concat |
| Neck/head feature joins | CPU upsampling, concatenation and layout conversions |
| Three detection heads | CPU convolutions, anchor distributions and decoded vectors |

Source-derived counts for one forward: **125 NPU Conv+BN+activation block
invocations**, comprising **60 1x1 GEMM and 65 multicore 3x3** operations.
The planner represents these as 54 repeated layer groups (32 GEMM, 22 conv).
The seven RepNCSPELANs contain 14 RepNCSPs and 42 bottlenecks; their 42 RepConv
modules contain **84 CPU Conv2d** operations. Detection adds **18 CPU Conv2d
and three Conv3d**. Thus the model contains 227 Conv2d plus three Conv3d, with
102 Conv2d remaining CPU. These counts describe placement, not test coverage.
See `model.py:39`, `layers.py:73`, `layers.py:132`, `layers.py:227`, and
`test_full_model_mc.py:94`.

Every graph block participates in the full-model comparison. That does not
exercise the separate NPU kernels for RepConv, complete Bottleneck/RepNCSP,
pooling, elementwise operations or complete SPP: the hybrid uses different
implementations for those pieces.

## Basic layer coverage

Each basic `run.lit` performs clean/all/run through its local Makefile, with
NPU2, Peano and Torch required. Each exercises one default configuration;
none sweeps the command-line options or all production shapes.

| Family | Default hardware lit shape/options | Maximum absolute error gate | Missing coverage |
| --- | --- | ---: | --- |
| Conv | 8x8, 8 to 8, 3x3, stride1/pad1, no activation | <0.10 | 1x1, stride2, full shapes, tiled/vector/fused variants not selected here |
| BN+SiLU | 8x8x8, combined | <0.25 | BN-only option, SiLU-only export, trained/nonidentity BN statistics |
| Elementwise | 512-element add | <0.05 | max/mul options and scalar-add export |
| AConv | 8x8, 8 to 8, pool plus downsampling conv | <0.25 | Production dimensions; standalone avgpool export |
| Bottleneck | 8x8, 8 to 8, residual enabled | <0.35 | Residual-off, channel and expansion variations |
| RepConv | 8x8, 8 to 8, stride1/pad1 | <0.30 | Production shapes/stride/padding combinations |
| RepNCSP | 8x8, 16 to 16, expansion0.5, one bottleneck | <0.40 | Production repeat3 and dimensional combinations |
| ELAN | 8x8, 32 to 32, part32/process16 | <0.45 | Full ELAN2 shape only in manual/composed or hybrid paths |
| SPP | 8x8, 16 to 16, neck8 | <0.15 | Full numerical resident SPP at production shape |

Evidence: each family's `Makefile` defaults and `test.py` comparison; notably
`conv/test.py:95,176`, `repncsp/test.py:73,221`, `sppelan/test.py:208`.

**No standalone lit entry exists for RepNCSPELAN.** It has monolithic and
streamed-weight generators plus monolithic, host-composed and tiled tests.
The tiled manual test uses 80x80, 128 to 128, repeat3 and a <0.5 final gate;
`elan/test_tiled.py` similarly covers 160x160 ELAN manually. Both rely on named
prebuilt artifacts. No test caller was found for
`repncsp_elan/aie2_streamed.py`. Historical monolithic memory failures are
documented in `repncsp_elan/IMPLEMENTATION_STATUS.md`, not freshly reproduced.

Several `make test` paths are misleading as correctness evidence: Bottleneck,
RepConv, RepNCSP, ELAN, SPP and RepNCSPELAN CPU branches only execute the
PyTorch reference and return success. The manual GEMM CPU path prints its
error and unconditional PASS without asserting a packing roundtrip or error
bound (`gemm_conv1x1/test_gemm_conv1x1.py:157`). Many original tests initialize
weights before seeding their input RNG, so their random weights are not
reproducibly seeded (for example `sppelan/test.py:67,72`).

## Fused, vector, batching and regime variants

The active consolidated kernel `kernels/rep_elan_bf16.cc` exports packed fused
1x1 and 3x3, fused GEMM, K-blocked GEMM, and residual-add/SiLU. Older scalar,
vector, partial-accumulation and composite sources remain alongside it.
There is no exhaustive generator-option x shape x kernel-symbol test matrix.

| Variant | Automated coverage | Limits |
| --- | --- | --- |
| Default fused conv/BN/SiLU, GEMM and K-blocked GEMM | Both full-model lit entries | One deployed input shape; final-output oracle; inactive alternatives not exercised |
| Standalone packed/tiled, multicore broadcast/PPC/chain variants | Primarily manual `conv/test_*.py` | Not individually enrolled in lit; some stale build mappings below |
| BO reuse / two-stage ObjectFIFO fusion | `run_fusion_primitives.lit`, 10 changing frames | Three small 8x8x16 tiles, distinct weights, same-context capability proof |
| WholeConv | `run_whole_conv.lit`, 48 cases | 12 trained re8/re21 weights x four inputs; exact baseline equality and 4-to-1 launches/readbacks |
| WholeGemm | `run_whole_gemm.lit`, 16 frames | ELAN2 projection 160x160x128 to 64; trained/changed weights, boundaries; exact 4-to-1 equivalence |
| WholeKBlocked | `run_whole_kblocked.lit`, 30 frames | re4/re15 projections 80x80x256 to 128, KB16; five inputs, exact 3-to-1 equivalence plus zero-row invariant |
| GEMM completion | `run_gemm_completion.lit`, seven methods | Generated IR, 32/5/1 cores, KB/non-KB and bounded batches; no arithmetic execution |
| Sparse K-block arithmetic | `run_kblocked_sparse_probe.lit`, 410 exact cases | Independent signed impulse/cancellation oracle, four rows/IC16/OC8/KB8; not full deployed KB16/OC128 envelope |
| Regime R1-R3, R5 and shared envelopes | Host routing/planner tests; historical manual hardware reports | No explicit full-model hardware lit route matrix |
| Persistent MDV6Executor | Mock lifecycle/benchmark tests | No lit numerical hardware run through persistent executor |

BO fusion checks independent arithmetic <0.05, finite results, bitwise route
equality, changing outputs and actual sync/launch counters
(`conv/test_bo_reuse.py:93`). The whole-operator tests similarly check finite
results and actual runtime counters, using explicit failures under `python -O`.
Their primary arithmetic oracle is the legacy NPU implementation, so a shared
kernel defect can pass both sides; the KB test explicitly documents this
(`gemm_conv1x1/test_whole_kblocked.py:2`). The sparse probe is the stronger
independent arithmetic check, but over a deliberately small envelope. Its
optional dynamic-extraction diagnostic is not required to pass by default lit.

**No lit file sets `MDV6_WHOLE_CONV_DIR`, `MDV6_WHOLE_GEMM_DIR`, or
`MDV6_WHOLE_KBLOCKED_DIR`.** Direct wrapper comparisons do not validate their
full-model composition, route combinations or persistent lifetime behavior.
Host route tests extract function ASTs and inject mocks. Likewise no lit
explicitly selects `USE_GEMM_CONV1X1=0` or the experimental regime routes.

Manual multicore checks vary in strength: `conv/test_multicore_conv3x3.py:69`
checks only four of 32 cores; batch/performance tests use identity BN and a
0.5 tolerance. Approximate-sigmoid references test consistency with the
approximation, not independently its difference from true SiLU. The script
named `test_model_dimensions.py:306` uses 8x8 stem examples at stride1, not
the deployed full-resolution stride2 stems.

Static stale build/ABI findings, not new compiler-run failures:

1. `gemm_conv1x1/Makefile:24` builds `gemm_conv1x1_bf16.o`, but
   `aie2_gemm_conv1x1.py:158` links `rep_elan_bf16.o`, with no corresponding
   rule in that Makefile. `--no-fuse` selects a symbol absent from the unified
   source. The old source's similarly named alias invokes the fused function
   and therefore expects BN parameters (`gemm_conv1x1_bf16.cc:79`).
2. `conv/aie2_tiled.py:111` requests `conv3x3_tiled_bf16` from the unified
   object; that symbol exists in `conv_bf16.cc:353`, while the tiled Makefile
   target only depends on `conv_bf16.o`.
3. The nonpacked path in `conv/aie2_fused.py:30` requests fused symbols absent
   from the unified object; the packed paths have the corresponding exports.

Default basic conv lit and the production builders take different paths and
do not protect these advertised manual entry points. Uncalled auxiliary
exports should be explicitly retired or enrolled, rather than counted as
tested merely because their source file also contains a tested function.

## SPP resident-fusion coverage

SPP contributes **117 of the 205 host unittest methods**, in 15 modules. Its
new host tests cover layout, persistent allocations, failure poisoning/no
retry, output ownership and exclusive diagnostic writes using mocked hardware.
IR tests reject mutated descriptors, locks, sizes, overlaps and completion
ordering. Many use authored minimal IR fixtures; these do not execute C++.

| Prototype | Host methods | Hardware lit workload | Claim supported |
| --- | ---: | --- | --- |
| Fusion schedule metadata | 6 | None | Storage/lifetime accounting; not executable IRON |
| Gather | 12 layout/host + 5 IR | 30 frames | Full-shaped opaque data gather/replication across four columns |
| Numerical phase-A shard | 11 host + 5 IR | 80 cases plus 17 pool cases | One eight-channel trained projection plus all three pools; all16 channel slices tested serially |
| Packet aggregate | 8 host + 5 IR | 30 frames | Four XOR workers to one memtile receive channel |
| Packet stripe join | 8 host + 5 IR | 30 frames | 25 bounded stripe joins with descriptor reuse |
| Packet plus gather | 9 host + 6 IR | 30 frames | Sixteen XOR workers aggregate and gather without host intermediate transfer |
| Phase alias | 11 host + 5 IR | 30 frames | One-core physical arena reuse, completion fences, finite A/25B phases and rearm |
| Numerical phase-A plus gather | 12 host + 9 IR | Feature-disabled | Intended 16 numerical workers plus gather; routing/compile feasibility unresolved |

The transport tests compare exact opaque bits using independent coordinate
oracles, multiple patterns and changing seeds, and enforce sync/run/load
counts. They are strong routing tests, not numerical SPP tests.

The numerical shard tests real projection/pooling arithmetic with all16
trained eight-channel slices x five input families. Projection is compared
bitwise with the existing production K-blocked NPU kernel; pooling is checked
against independent CPU pooling. Seventeen pool-only cases cover negative
padding, borders/corners, channels, random, zero and signed-zero ties.
Projection arithmetic shares implementation with the baseline, so its oracle
has the same common-defect limitation. See `test_phase_a_shard.py:94,109,129`
and `phase_a_host.py:58`.

`run_spp_phase_a_gather.lit:7` requires `mdv6_phase_a_gather_routing`, which
no configuration enables. The implementation and host/IR tests exist, but no
executable/hardware validation is established. Its checker explicitly reports
`hardware_approved=False` (`check_phase_a_gather_ir.py:400`). Existing follow-up
issues `mlir-aie-2vb.2.4` and `mlir-aie-8cu` track integration/routing work.

Full numerical phase-B projection, the complete multi-worker phase barrier,
final output joins and full-island rearm remain integration work. The one-core
alias proof does not establish these. See
`sppelan/PHASE_A_GATHER_VALIDATION.md:298,327`.

Validation documents record 30/100/300-frame campaigns for transport/alias
probes and 80/100/300 for the arithmetic shard. These are historical artifact
results; normal lit does not reproduce all those long campaigns.

## Full-model oracle and runtime gaps

`run_full_model.lit` runs trained BF16 inference on one random
`[1,3,640,640]` input, seed42. `detection_validation.py:5` requires exactly
three scales, three tensors per scale, matching shapes and finite values in
both reference and actual, including anchors. Numerical comparison covers
only class logits and decoded vectors, **each max error <5.0**. Finite but
incorrect anchor distributions are not directly error-gated. Layer debug
prints are not mandatory intermediate feature assertions.

`run_full_model_multi_frame.lit` runs `--profile 3` and then three changing
seeds through `validate_stream.py`, with tighter **class <0.5/vector <0.1**.
Profiling itself repeats seed42; streaming changes seeds42/43/44 but rebuilds
the model each frame. The validator rejects diagnostic shortcuts, flushes
failure reports and stops on first failure. RSS, contexts and launches are
recorded, without bounded leak/resource regression assertions.

Other material gaps:

- Final CPU reference and hybrid CPU portions reuse the same layer code;
  agreement does not independently establish CPU model/weight-mapping fidelity.
- `DEBUG_GEMM_TRAINED` makes the ordinary full-model entry return success
  before a graph forward (`test_full_model_mc.py:190`). Other debug modes also
  substitute diagnostics. Lit does not explicitly clear these or route flags.
- Both full-model lit builders use `build_multicore.py || true`, accepting a
  known unused overflow but also masking unrelated build failures. Required
  missing artifacts fail at runtime; preferred PPC variants can fall back
  (`run_tiled_mc.py:291`), so success does not prove every variant built.
- Host tests cover executor, buffer contracts, metrics, routing and planner
  well, but use mocks. Legacy pack-cache identity/lifetime behavior, BO pools,
  retry/reload and variant fallback have little direct behavioral coverage.
- `WholeConv.run` ignores unsuccessful result status and lacks the explicit
  failure handling tests present for WholeGemm/KBlocked (`whole_conv.py:89`).
  Its direct hardware test's completed-run counter can detect unsuccessful
  returns, but that is not equivalent to a runtime failure contract.
- No lit passes the profiler `--baseline` option. Its optional gate checks
  category regressions >10% only when baseline category >=5ms, omits total
  wall time and launch count, and silently skips a missing baseline path
  (`profile_harness.py:385`). Planner launch-count assertions validate a cost
  model, not observed runtime performance.
- No labeled-image detection quality, mAP/precision/recall, preprocessing,
  postprocessing/NMS or production data distribution gate is enrolled here.
  Numerical agreement on random inputs is a narrower claim.
- Legacy `test_full_model.py` is not enrolled in lit and has weaker gates and
  random model initialization than the active multicore test.

## Shared CPU reference and support package

Outside the inventory counts, `../../../python/mdv6` contains 11 layer classes,
the full model, BF16 wrapper, checkpoint mapper, inference/preprocessing and
decode/NMS/visualization support. These are part of the effective model surface.

- `test_mdv6.py` runs random 320x320 and 640x640 forwards and prints shapes,
  without numerical or shape assertions. Its standalone main converts a
  returned failure Boolean to nonzero exit; pytest would not turn the caught
  exception's returned `False` into an assertion failure.
- `test_against_reference.py` attempts the independent PytorchWildlife
  implementation, but only prints numerical differences, including large
  failures (`:287`). Missing weights return normally; forward exceptions are
  caught; main does not propagate a failing exit status (`:299`). Consequently
  it is a diagnostic, not an independent enforced oracle. Its expected local
  `python/camera_traps_reference` directory is absent in this checkout.
- No focused regression tests were found for checkpoint mapping/completeness,
  the BF16 wrapper, preprocessing, decode or NMS in this package. Mapper and
  BF16 loading use `strict=False`; missing/unexpected keys are not an enforced
  compatibility contract. Separate `detection_validation.py` tests in this
  example check numerical comparison logic, not those support operations.
- Static support-code defects illustrate the gap: `run_inference.py:16`
  imports `model_bf16` as a top-level module although `model_bf16.py:5` requires
  a package-relative import; `keep_bn_fp32` is accepted but the conversion loop
  converts all FP32 parameters/buffers (`model_bf16.py:15,54`). These were found
  by source inspection, not by executing those entry points in this audit.

## Discovery, CI and coverage measurement

`../../../programming_examples/lit.cfg.py:28` discovers only `.lit`, so a
`test*.py` filename alone is not test registration. The CMake target is
`check-reference-designs` (`programming_examples/CMakeLists.txt:134`).

`mdv6/lit.local.cfg:8` marks the whole subtree unsupported without AIE2P,
including CPU-only tests. Parent `ml/lit.local.cfg:10` additionally requires
AIE2; AIE2P-only configurations can inherit that unsupported state. This
checkout has both, so its two hardware-free lit gates run. Ordinary host-only
CI can omit all of MDV6 despite being able to run the unittest suite directly.
Trained weights are an untracked staged prerequisite; lit adds `mdv6_weights`
only if the adjacent file exists.

No MDV6-specific test invocation was found in the inner mlir-aie repository's
GitHub workflows. **The outer npu-dev-mdv6 repository does have a dedicated
workflow**, `.github/workflows/mdv6.yml:107`: it builds with AIE2 and AIE2P,
stages cached trained weights when available, and invokes
`lit programming_examples/ml/mdv6 -v` without a test filter. It runs for
pushes/PRs to the mdv6 branch on a self-hosted runner. This is real CI wiring
for this suite, including new discoverable entries; this audit did not inspect
successful CI run logs. Missing cached weights allow weight-dependent tests
to report UNSUPPORTED rather than fail, with no required pass-count gate.

The outer workflow also runs a `--profile 3 --baseline` step, but marks it
`continue-on-error: true` and skips it when weights/build artifacts are absent
(`mdv6.yml:140`). Performance regression is therefore a soft CI signal.
Its later `pytest mdv6/test_mdv6.py` step targets a separate outer `mdv6`
submodule/container, not this inner repository's `python/mdv6` package.

The inner repository's C/C++ coverage job instruments `aie-opt` and depends on
`check-aie` (`../../../CMakeLists.txt:371`), rather than measuring the Python
MDV6 runtime and separately compiled AIE kernels. No MDV6 pytest-cov,
coverage.py, kernel instrumentation or enforced coverage percentage was found.
Lit registration proves discoverability, not routine CI execution or passing
hardware results.

## Follow-up priorities

1. **Make enrollment reliable:** portable host gates, explicit baseline route,
   diagnostic rejection and required CI execution/pass-count checks. Filed `mlir-aie-7c7`.
2. **Declare supported variants and repair their clean builds:** resolve old
   object/symbol mismatches, enroll RepNCSPELAN and retained manual variants,
   sweep representative production shapes/options. Filed `mlir-aie-pf1`.
3. **Strengthen independent correctness and integration:** intermediate
   features, anchor error checks, supported whole/regime route combinations,
   persistent changing-frame inference. Filed `mlir-aie-a46`.
4. **Cover runtime failure/cache behavior and measurable regressions:**
   WholeConv failures, lifetime/reload paths, explicit optional performance
   and memory gates. Filed `mlir-aie-iui`.
5. **Finish SPP numerical integration before claiming full resident fusion:**
   existing `mlir-aie-2vb.2.4` / `mlir-aie-8cu` cover the blocked next step;
   full phase-B/barrier/output integration remains beyond that step.
6. **Enforce independent model fidelity and CPU support behavior:** make the
   official-reference comparison fail on errors, test checkpoint mapping and
   BF16 contracts, and cover retained inference support. Filed `mlir-aie-noa`.

The observed coverage is broad but uneven: default graph execution has a
numerical smoke gate and a short stronger stream gate; newer bounded routing
and scheduling primitives have detailed behavioral tests; all-layer,
all-variant, independently checked production numerical coverage is absent.

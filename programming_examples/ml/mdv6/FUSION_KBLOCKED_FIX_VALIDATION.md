# GEMM completion and row-extraction fixes — 2026-09-09

This is the corrective follow-up to [FUSION_KBLOCKED_VALIDATION.md](FUSION_KBLOCKED_VALIDATION.md),
based on `3b6b48266ff97e2363e3966b7fe75e13aa8e8c21`. That earlier document
records the failure state, not the current generator or strict-test behavior.
The commit containing this document contains the corrected implementation.
No experimental batching route is enabled by default.

Bounded tasks `mlir-aie-2vb.6` (correctness fixes) and `mlir-aie-2vb.5`
(K batching) are closed after the gates below. Full-island work remains in
`mlir-aie-2vb.2`; the installed compiler still needs upstream follow-up in
`mlir-aie-2vb.7`.

Two independent defects were isolated and addressed: incomplete cross-column
completion in the original GEMM runtime, and incorrect dynamic extraction of
eight-lane rows from an otherwise correct 32-lane MMUL result on the installed
toolchain. All selected K-blocked trained-weight tests now pass exact equality
**and** the previously failing zero-input-row invariant. Shared-kernel changes
require fresh convolution and GEMM artifacts; old xclbins do not acquire the
fix by updating Python or setting an environment variable.

## Causal evidence and implementation

### Every independent output column must finish

The old unbatched GEMM awaited only column 7. Sparse-boundary inputs exposed
stale output in column 0: exactly its first 272 pixels in the selected shape.
A newly compiled unchanged reference reproduced the mismatch. A reference
whose IR changed only output completion tokens/awaits matched the batched route
over 30 cases. This is a completion dependency, not a numerical tolerance issue.

`gemm_conv1x1/aie2_gemm_conv1x1.py` now awaits **every output column**, for
K-blocked, non-K-blocked, partial-column and regime-generated programs alike.
The selected unbatched reference has eight awaits; three-batch K GEMM has 24.
Inputs/weights are freed only after output completion. Whole-operator groups
remain bounded: the next group is configured after prior fills are freed.
No additional groups are left simultaneously outstanding by this fix.

`wait_all_columns=True` and `--wait-all-columns` remain compatibility interfaces;
explicitly disabling the Python argument is rejected. The old build-time
`MDV6_GEMM_WAIT_ALL_COLUMNS` variable is unnecessary: all values now leave safe
completion enabled. This intentionally changes default unbatched generated IR.
Historical unsafe behavior should be reproduced from the old commit/artifacts,
not by adding an unsafe switch to production dispatch.

### Correct MMUL result, incorrect dynamic row extraction

The isolated probe calls the production KB8 kernel on four rows, IC16/OC8.
It exposes the first K partial, final result, assembled/transposed A operand,
all eight broadcast operands, imported partial, a direct MMUL result, raw
float32 accumulator, dynamic row stores and literal-index row stores.

The causal v2 observation was that the full 32-lane direct result and raw
float32 accumulator were correct, while storing `result.extract<8>(row)` inside
a runtime-indexed loop copied a negative fourth-row lane into another row.
Operand assembly and broadcast traces passed. This distinguishes extraction
from a wrong matrix product. Do not attribute an imported-partial mismatch
solely to accumulator import without comparing it with the *observed* upstream
partial, rather than only the ideal oracle.

`kernels/mmul_bf16_rows.h` invokes a templated lambda at literal row indices
0, 1, 2 and 3. Eight output loops use it: four in the unified kernel, three in
the standalone GEMM duplicate, and one in the standalone convolution duplicate.
MAC ordering, first-K initialization, partial bf16 storage boundaries,
last-K-only BN/SiLU, and bounds guards are unchanged. Both final and intermediate
K outputs are covered. The helper is an explicit workaround for the observed
installed-toolchain behavior, not a new mathematical approximation.

The probe intentionally retains dynamic extraction as a separately reported
compiler diagnostic. It is excluded from the production-correctness gate by
default; `--check-dynamic-extract` makes any observed diagnostic mismatch
fail-stop. The failure is code-generation-context-sensitive: the historical
v2 artifact failed, but v5 reported zero dynamic-extraction failures. V6 keeps
the dynamic loop in a separate noinline function with loop unrolling disabled;
it reports diagnostic failures in 154 of 410 cases while every production and
literal-index gate passes. Merely enabling the flag does not guarantee a
reproduction on another compiler/context. Production partial/final results
and literal-index extraction remain strict.

### Probe corrections and numerical contract

An early probe packed each logical KB8/OC8 weight chunk into 80 bf16 elements.
Its second chunk began at byte 160, violating the 64-byte alignment required
by its 64-lane weight load. The current probe uses 96 elements per chunk:
64 weights, 16 BN fields and 16 zero padding. A compile-time alignment check
and CPU round-trip/padding checks enforce this. The selected production
KB16/OC128 chunk is already 2304 bf16 elements and was not misaligned.
Do not treat the early probe's second-chunk discrepancies as production defects.

The final probe ABI is input 64, weights 192, output 704 uint16 elements.
Output elements 512..575 encode 32 float32 accumulators and are compared as
float32, not as two bf16 halves. Elements 672..703 are raw metadata; the first
records `get_rnd()` and all are compared as integers. No rounding-mode write is
introduced. The oracle uses floor conversion and requires observed `rnd_floor=0`,
with independent hardcoded positive/negative midpoint and scalar-SiLU checks.
The final v6 hardware run passes all 410 independent production/literal cases
with observed floor mode 0, while reporting the 154 dynamic-stage failures
separately. CPU construction checks also pass; the hardware observation, not
those CPU checks alone, establishes the rounding mode for this run. V5 also
passed the independent gate, but its dynamic diagnostic happened to pass in
that code-generation context, as distinguished above.

## Fresh builds and resource gates

Full corrected artifact root: `/tmp/mdv6-full-fixed.jGu5Ua`.
Temporary paths are evidence locations, not permanent installation contracts.

- GEMM: all 18 configurations built successfully, zero failures, 93 seconds.
  All 720 tile allocation maps are bounded/nonoverlapping. Across 576 core
  ELFs, maximum text is 13760 bytes; data/BSS are zero.
- Convolution: the current script enumerates 49, not the historical 50.
  It built 48 successfully; `mc_ftconv0_p4` remains the known unused oversized
  configuration failure. Do not hide the nonzero build result or describe this
  as every enumerated convolution building. Production full-model runs below
  prove the required routes are present.
  All 1920 tile maps and 1536 core ELFs were inspected; maximum text is 12640
  bytes, data/BSS are zero, maximum L1 allocation ends at 61676 and maximum
  memtile allocation ends at 419968 bytes.
- Whole GEMM, whole convolution and whole K GEMM all built successfully.
  Each has 40 valid tile allocation maps; stack reserves are respectively
  8192, 4096 and 8192 bytes. Core-0 text is 12448, 12752 and 13712 bytes;
  output-await counts are 32, 32 and 24.
- Selected K baseline/whole text grows from 8304/8352 to 13664/13712 bytes.
  Its L1/L2 layouts are unchanged: L1 metadata ends at 65068, leaving only
  468 bytes below 64 KiB; memtile allocation ends at 208896. Do not add buffering
  without another placement and instruction-size gate.

Header dependencies were added to the shared/standalone/experimental Makefiles.
GEMM artifact reuse now requires xclbin, instructions and MLIR, plus fresh
object, generator, kernel source and local headers. Missing dependencies no
longer count as up-to-date. The main build intentionally recompiles its kernel
object, forcing selected artifacts to regenerate even in a reused build tree.
The generated-IR test includes missing/stale artifact cases.

Compiler: clang 21.0.0, LLVM-AIE revision
`9e603b765b27cae1566a02965eb0152640199850`, installed under
`/home/jfifield/npu-dev-mdv6/venv/lib/python3.12/site-packages/llvm-aie/bin`.
Recorded SHA-256 values:

```text
unified source 354d663b875fad8b591190149ff9b30712445c1b0c6645269f8682c3c4b4f928
row helper     78c9c8cdd74fbf88990e321a6ce6424c1136838495c10204cb5c4403dcf9f8b6
GEMM object    5f9335339e8a4dd2836f20fdb90993d18a1dad3aa22964466c0c8946712297ec
trained weights 4d0af6e3d80bbbdbcc22a5ec6cf997f66e3f6062e463a04c1da34cf08f44c3ae
```

## Validation checkpoint

| Corrected route/check | Result |
|---|---|
| Selected K baseline versus batched, both trained projections | 30 cases PASS, exact bits and zero-row invariant |
| Whole non-K GEMM versus fresh baseline | 16 cases PASS, exact bits |
| Whole convolution versus fresh baseline | 48 cases PASS, exact bits |
| Independent aligned sparse probe v6 | 410 production/literal cases PASS, observed rounding mode floor (0); 154 separately reported dynamic-stage failures |
| Fresh default full model | 3 and 30 frames PASS, 453 submissions/frame |
| Fresh K-batching-only full model | 3 frames PASS, 449 completed submissions/frame |
| Fresh combined three batching routes | 3, 30, 100 and 300 frames PASS, 410 completed submissions/frame |
| Completion/freshness CPU and generated-IR tests | 7 PASS, also under `-O` |
| Configured `run_gemm_completion.lit` | PASS |
| Full CPU suite | 101 tests PASS |
| Standalone convolution and GEMM source copies | Both compile with installed Peano at `-O2` |
| Configured host/completion/K/GEMM/conv/sparse lit gates | All 6 PASS, hardware jobs serialized |

The short full-model runs use seeds 42–44, finite outputs and class/vector
thresholds 0.5/0.1. Warm default uploads/downloads are 810/453 calls and
424077184/87130112 bytes; combined counts are 731/410 and
409331584/87130112 bytes. These are host-runtime observations, not device-DMA
counters. Intermediate activations still return to host. Three frames provide
integration evidence, not a latency ranking or sustained promotion gate.

The corrected baseline and combined routes passed 30 frames each; combined also
passed 100 and 300. These runs use the freshly
rebuilt shared kernels, not earlier artifacts. Default route selection remains
unchanged. In every completed stage, cold context-cache misses are 32, with no
warm misses or evictions. This does not measure driver-level context switches.
No real-image detection-quality, indefinite-memory or timeout-recovery claim
is made by the exact synthetic/operator gates.

Matched fixed-order measurement (default then combined, 30 frames each, seeds
42..71, no compilers active): mean over the 29 warm frames is 1569.836 ms versus
1534.182 ms; medians are 1570.108 versus 1530.182 ms. That is an observed 35.655 ms
(2.27%) reduction, not a repeated randomized performance ranking. Both have
maximum class/vector differences 0.233887/0.03125. The separate combined
100-frame run averages 1532.731 ms warm, with maxima 0.238281/0.03125.
The 300-frame run averages 1535.681 ms warm and has the same maxima; every frame
passes, with 410 completed submissions and zero warm cache misses/evictions.
No old-artifact timing is used as a correctness-fixed control.

Batching removes 43 of 453 submissions (9.49%) and 14,745,600 host-upload bytes
per frame (3.48%), but downloaded bytes and cached context count are unchanged.
This is a modest enabling gain, not inter-operator device residency. Retain
the opt-in routes while implementing the fused-island plan below.

Evidence logs: `/tmp/mdv6-fixed-production-30.jsonl`,
`/tmp/mdv6-fixed-whole-gemm-16.jsonl`, `/tmp/mdv6-fixed-whole-conv-48.jsonl`,
`/tmp/mdv6-fixed-default-3.jsonl`, `/tmp/mdv6-fixed-combined-3.jsonl`,
`/tmp/mdv6-fixed-konly-3.jsonl`,
`/tmp/mdv6-fixed-default-30.jsonl`, `/tmp/mdv6-fixed-combined-30.jsonl`,
`/tmp/mdv6-fixed-combined-100.jsonl`, `/tmp/mdv6-fixed-combined-300.jsonl`,
`/tmp/mdv6-sparse-v6.jsonl`, `/tmp/mdv6-fixed-lit.log`,
`/tmp/mdv6-fixed-sparse-lit.log`. Enabling `--check-dynamic-extract` on the
frozen v6 artifact fails the first signed-fourth-row case solely in the unsafe
dynamic stage (output index 480); see `/tmp/mdv6-sparse-v6-dynamic.jsonl` and
`/tmp/mdv6-sparse-v6-dynamic-failure/`. This is a retained compiler reproducer,
not a failing production gate. Upstream follow-up is `mlir-aie-2vb.7`.
The full-model timer includes hybrid inference, layouts, CPU islands and
detection, but excludes construction, CPU reference and comparisons.

## Reproduce from a fresh context

Read this document, [FUSION_PERF_PLAN.md](FUSION_PERF_PLAN.md), and README.
Inspect the relevant beads and preserve unrelated user files. Use new artifact
directories; never mix pre-fix operator artifacts into a claimed corrected run.

```bash
source /home/jfifield/npu-dev-mdv6/env.sh
export PYTHONPATH=/home/jfifield/npu-dev-mdv6/install/mlir-aie/python:${PYTHONPATH:-}
export MDV6_REGIME_ROUTE=legacy USE_REGIME_XCLBINS=0 USE_REGIME_KBLOCKED=0 USE_GEMM_CONV1X1=1
unset MDV6_WHOLE_CONV_DIR MDV6_WHOLE_GEMM_DIR MDV6_WHOLE_KBLOCKED_DIR
fixed_build=$(mktemp -d /tmp/mdv6-fixed.XXXXXX)
export MDV6_BUILD_DIR="$fixed_build"
python3 gemm_conv1x1/build_gemm_conv1x1.py
# Expected known unused mc_ftconv0_p4 failure; inspect all results explicitly.
python3 conv/build_multicore.py
make -C "$fixed_build" -f "$PWD/gemm_conv1x1/Makefile.whole_gemm"
make -C "$fixed_build" -f "$PWD/gemm_conv1x1/Makefile.whole_kblocked"
make -C "$fixed_build" -f "$PWD/conv/Makefile.whole_conv"
make -C "$fixed_build" -f "$PWD/gemm_conv1x1/Makefile.kblocked_sparse_probe"
python3 -O test_gemm_completion.py
python3 -O gemm_conv1x1/test_kblocked_sparse_probe.py --build-dir "$fixed_build" --cpu-only
```

Inspect compiled allocation maps and stop other compilers before timing.
Run NPU jobs serially, with no retry after a device error:

```bash
python3 -O gemm_conv1x1/test_kblocked_sparse_probe.py --build-dir "$fixed_build"
python3 -O gemm_conv1x1/test_whole_kblocked.py --build-dir "$fixed_build" --frames 30
python3 -O gemm_conv1x1/test_whole_gemm.py --build-dir "$fixed_build" --frames 16
python3 -O conv/test_whole_conv.py --build-dir "$fixed_build" --frames 48
python3 benchmark_executor.py --frames 3 --report "$fixed_build/default-3.jsonl"
export MDV6_WHOLE_CONV_DIR="$fixed_build"
export MDV6_WHOLE_GEMM_DIR="$fixed_build"
export MDV6_WHOLE_KBLOCKED_DIR="$fixed_build"
python3 benchmark_executor.py --frames 3 --report "$fixed_build/combined-3.jsonl"
```

To repeat the completed sustained campaign, advance through 30, 100 and 300
changing-input frames only after each stage passes, recording matched timing
and memory/context observations. Preserve
failure tensors and provenance instead of widening tolerances. After these
correctness fixes, resume the device-resident fusion roadmap; command batching
alone does not remove inter-operator activation traffic.

## Next performance cut: SPP gather-only proof

Return to `mlir-aie-2vb.2` and `sppelan/FUSION_SCHEDULE.md`. Prove transport
before adding convolution, pooling or phase-buffer aliasing:

1. Add a gather-only generator/kernel/build/test under `sppelan/`. Agree on
   raw uint16 input `[source_column4,row4,level4,pixel400,lane8]` (409600 bytes)
   and inspected output `[destination_column4,stripe25,pixel16,K512]`
   (1638400 bytes). Logical K order is level then neck channel. Reuse the
   existing `gather_segments()` mapping as the host ordering oracle.
2. First compile one remote-column transfer, then all four sources/destinations
   and 25 stripes. Retain four 100-KiB source memtile regions. Use bounded source
   rounds and one destination stripe slot initially, with explicit completion
   before descriptor/buffer reuse; do not issue 64 concurrent segment tasks.
3. Prefer direct memtile routing if supported. A relay-worker alternative needs
   an explicit revised schedule: 16-KiB stripe plus 256-byte segment storage,
   and an account of how relay workers fit the eventual 16-worker budget.
4. Test two raw-bit frames encoding low/high halves of each source element's
   index, then changing seeded-random frames. A single uint16 value cannot
   uniquely identify all 204800 source elements. Require bitwise correct output
   at all four destinations, bounded compiled resources, all-column completion,
   one submission, and only the initial upload/final diagnostic download.

This proves routing, not full SPP fusion. The 1.6-MiB gathered traffic, including
1.2 MiB across columns, is schedule-derived traffic, not a bandwidth measurement.
Next prove phase-L1 aliasing and one 20x20x8 projection/pooling shard before
connecting the island. No additional production contexts or host repacking
should be introduced without an explicit contract and measured justification.

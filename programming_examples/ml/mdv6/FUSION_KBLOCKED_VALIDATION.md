# K-blocked spatial batching — 2026-09-09

## Status and acceptance boundary

Implemented, built and opt-in; **not accepted for promotion**. The selected
operator now uses one submission instead of three. It is bitwise equivalent
over 30 changing-input cases to an otherwise unchanged baseline with explicit
all-column completion. Short full-model integration passes the existing
tolerances at 449 submissions, or 410 with both prior batching experiments.

The stronger boundary tests uncovered two correctness defects. Equivalence to
the old kernel is not sufficient mathematical validation. Task `mlir-aie-2vb.5`
remains incomplete; correctness follow-up is `mlir-aie-2vb.6` (P1). Do not enable
this route by default or describe the latency numbers below as an accepted win.

## Exact scope and ABI

`model.rep_elan4.conv4` and `model.rep_elan15.conv4` both use `gemm_re4_c4`:
80x80, IC256, OC128, 32 cores, tile M68, PPC1, K16 (16 chunks). Only this
name and shape route through `MDV6_WHOLE_KBLOCKED_DIR`; unset means unchanged
default. Input, weight and output arenas persist; packing refreshes each call.
Errors invalidate the executor without retry or fallback. Outputs own their
host storage. There is still no inter-operator activation residency.

| Buffer | Layout | Bytes |
|---|---|---:|
| Input | uint16 bf16 `[3,32,1,68,256]` | 3,342,336 |
| Output | uint16 bf16 `[3,32,1,68,128]` | 1,671,168 |
| Packed weights | uint16 `[16,2304]` | 73,728 |

Raw weights are OI128x256 plus 128 BN scales and 128 biases (33,024 elements).
Each K16 chunk is packed `[KB/8,OC/8,innerIC8,innerOC8]` and followed by repeated
BN fields. All sixteen chunks replay for each spatial patch. Neither host
transfer bytes nor device weight traffic are reduced by this increment.
The last batch has 2,048 pixels: 30 full slots, eight real rows plus 60 zero rows
in slot 30, and a duplicate of slot zero in unused slot 31. Output crops to 6,400.

The worker loops over three spatial patches with the original unrolled K starts
0,16,...240. Kernel source, initial zeroing, intermediate bf16 partial stores,
and last-chunk-only BN/SiLU are unchanged. Three bounded runtime groups each
issue eight weight, eight input and eight output tasks, await all eight outputs,
then free fills. Compiled totals: 72 configurations, 24 awaits, 48 fill frees.

All 40 tile allocation maps were checked for overlap/bounds. Compute-tile input
occupies 8192..43008, output 43008..60416, chunk weights 60416..65024, RTP
65024..65048, barrier 65056..65068. Only 468 bytes remain above metadata below
64 KiB; do not add double buffering casually. Memtile allocation ends at 208896.
Default K, default non-K and existing four-batch non-K generated IR were verified
byte-identical before/after the batching change.

## Discovered defects and causal evidence

1. **Legacy completion:** the original generator awaits only column 7. The
   zero/negative/cancellation/random cases first passed; frame 8 (re4 boundary
   impulses) disagreed on exactly the first 272 pixels, column 0 of batch 0,
   maximum difference 7.78125. A fresh unchanged baseline reproduced the same
   failure. The new diagnostic `--wait-all-columns` generator switch changes
   only completion handling (eight awaits instead of one). A fresh build with
   it matches batching bitwise across 30 cases. Default generation is still
   unchanged; rolling out the completion fix requires review and affected builds.
2. **Shared sparse-input defect:** with zero BN biases and finite weights, zero
   input rows must produce zero regardless of MAC reduction order. Both routes
   violate this. Saved re4 evidence has five leaked rows in batching:
   65,2173,4349,6389,6397, maximum magnitude 5.25. Four outside column 0 are
   identical in the original baseline. Each is two rows before a negative
   impulse. The original baseline additionally returns varying stale nonzeros
   on zero rows in column 0. The mechanism of the shared leakage is **not yet
   diagnosed**; matrix lane/partial-sum mapping is a hypothesis, not a conclusion.

The hardware test always reports the zero-row invariant on boundary cases.
`--check-zero-rows` makes it a strict, currently failing semantic gate and
`--failure-dir` saves input, raw weights and both output bit patterns in an
exclusive NPZ file. Default test PASS is explicitly *legacy equivalence*, not
mathematical correctness. No tolerances were widened or failing cases removed.

Final strict run against the fenced reference failed as intended at frame 8:
both routes leaked on five of 6,390 zero-input rows (640 output elements,
maximum 5.25), despite exact equality between routes. Evidence was saved under
`/tmp/mdv6-kblocked-semantic-failure/`; log `/tmp/mdv6-kblocked-semantic.jsonl`.

## Evidence and timing limits

The isolated test alternates the actual retained re4/re15 weight arrays and
zero, negative, cancellation, random and batch-boundary inputs. Legacy retries
are disabled. Every successful comparison checks finite output, exact uint16
bits and completed dispatch/readback counts (3 versus 1, zero reloads).
30 cases against the fenced reference passed. Warm operator means over cases
1..29 were 18.268 ms reference versus 15.973 ms batching, alternating order.
Packing, transfers and readback are included; initialization is excluded. These
are diagnostic timings, not a sustained full-model performance claim.

| Short integration, 3 frames | K batching only | All three batching routes |
|---|---:|---:|
| Completed submissions/frame | 449 | 410 |
| Warm uploads/frame | 806 | 731 |
| Warm upload bytes/frame | 424,077,184 | 409,331,584 |
| Downloads/frame | 449 | 410 |
| Download bytes/frame | 87,130,112 | 87,130,112 |
| Warm inference mean, only 2 samples | 1563.934 ms | 1527.088 ms |
| Maximum class/vector differences | 0.226929 / 0.03125 | 0.226929 / 0.03125 |

Both had 32 cold context-cache misses and zero warm misses/evictions, with 20
warm load calls. Tolerance pass is not evidence against the exact zero-row bug.
No matched full-model baseline timing, 30/100/300-frame full-model campaign or
default promotion was attempted after discovering the semantic defect.

Environment: Strix Halo NPU2, Linux 6.17.0-20-generic, installed runtime under
`/home/jfifield/npu-dev-mdv6/install/mlir-aie/python`, weights SHA256
`4d0af6e3d80bbbdbcc22a5ec6cf997f66e3f6062e463a04c1da34cf08f44c3ae`.
Changes are based on `fc6709ea6`. Runtime device-DMA, cycle, driver-switch and
host-wait counts remain unavailable, not zero.

## Reproduce in fresh context

Source `env.sh` and set installed `PYTHONPATH` as in README. Build into new
directories, never replace production artifacts with the diagnostic reference:

```bash
kb_build=$(mktemp -d /tmp/mdv6-kb.XXXXXX)
kb_reference=$(mktemp -d /tmp/mdv6-kb-reference.XXXXXX)
make -C "$kb_build" -f "$PWD/gemm_conv1x1/Makefile.whole_kblocked"
MDV6_BUILD_DIR="$kb_reference" MDV6_GEMM_WAIT_ALL_COLUMNS=1 \
  python3 gemm_conv1x1/build_gemm_conv1x1.py gemm_t68_ic256_oc128_kb16_p1
MDV6_BUILD_DIR="$kb_reference" python3 -O gemm_conv1x1/test_whole_kblocked.py \
  --build-dir "$kb_build" --frames 30
# Independent semantic gate: currently expected to fail, do not suppress it.
MDV6_BUILD_DIR="$kb_reference" python3 -O gemm_conv1x1/test_whole_kblocked.py \
  --build-dir "$kb_build" --frames 10 --check-zero-rows \
  --failure-dir "$kb_build/failure"
```

Omit `MDV6_GEMM_WAIT_ALL_COLUMNS=1` in another fresh reference directory to
reproduce the legacy completion mismatch. The variable acts at **build time**;
it does not fix prebuilt artifacts. `run_whole_kblocked.lit` rebuilds both
artifacts and checks fenced legacy equivalence; `run_host_checks.lit` includes
the packing, failure, routing and diagnostic CPU tests.

For short integration set `MDV6_BUILD_DIR` to the full existing build root and
`MDV6_WHOLE_KBLOCKED_DIR` to the experimental directory before Python imports.
Use `benchmark_executor.py --frames 3 --report <new-path>` and baseline flags
from README. Prior experimental directories may be enabled to reproduce 410.
Unset all three whole-operator directory flags for default comparisons.

Temporary evidence this session: `/tmp/mdv6-kblocked-failure-2/` (original
mismatch NPZ), `/tmp/mdv6-kblocked-fenced-operator.jsonl`,
`/tmp/mdv6-kblocked-model-3.jsonl`, `/tmp/mdv6-kblocked-combined-3.jsonl`.
Builds: `/tmp/mdv6-whole-kblocked.hJZhl5`,
`/tmp/mdv6-kblocked-baseline.ufXOE8`, `/tmp/mdv6-kblocked-fenced.cMsvMZ`.
These may disappear; the scripts and this summary are the durable handoff.

Quality gates: 94 CPU tests pass; configured host lit and fresh-build K-blocked
hardware equivalence lit both pass. The independent strict semantic gate fails
as documented above, and remains an acceptance blocker.

## Next actions

1. Claim `mlir-aie-2vb.6`; reduce the sparse negative-impulse failure to one
   worker and selected channels/K chunks. Preserve exact zero-row oracle and
   trained-weight reproduction; do not use legacy equality as the sole gate.
2. Inspect mmul input/output lane and accumulator import/export mapping, then
   test a causal fix independently of completion fencing. No kernel change was
   made in this increment. Stop on any hardware timeout; do not auto-retry.
3. Review explicit completion for default GEMM and rebuild affected artifacts.
   Repeat sparse and dense tests for K/non-K and existing batching routes.
4. Once both defects are resolved, rerun K batching's isolated strict gate,
   matched full-model latency/counters, and 30/100/300-frame acceptance before
   promotion. Then resume device-resident SPP gather proof and K-weight traffic
   reduction; command batching alone is not the fusion goal.

# Whole-spatial GEMM batching — 2026-09-09

The next Milestone 1 increment batches ELAN2's final 1x1 projection into one
submission. It is opt-in and composes with the previous
[whole-convolution route](FUSION_M1_VALIDATION.md). The default route and saved
baseline are unchanged. Work began at `43e3621e0`; the commit containing this
document contains the tested implementation.

## Implemented capability

ELAN2 Conv4 is `160x160x128 -> 160x160x64`. Preserve the deployed non-K-blocked
GEMM: 32 cores, 104 pixels/patch, two patches/core, full IC/OC per patch.
Each original submission handles 6656 pixels; four cover 25600 real pixels.
The coherent program handles all four batches in one submission, with the
original mmul, BN and approximate-SiLU kernels and rounding boundaries.

Each worker acquires its weights once, processes eight patches, then releases
weights and the invocation barrier. This proves within-operator on-chip weight
reuse across four spatial batches; it does not retain weights across frames or
retain intermediate activations between different operators. K-blocked GEMM
batching is explicitly rejected for now to avoid silently changing its bf16
partial-sum semantics.

`gemm_conv1x1/aie2_gemm_conv1x1.py` adds optional `spatial_batches=1`.
Default generated IR was verified byte-identical to the previous source for
the selected non-K configuration and a representative K-blocked configuration
(`tile24, IC512, OC256, KB32, PPC1`). The wrapper/Makefile build a separate
`whole_gemm.xclbin` and instruction stream; no binary concatenation or baseline
artifact replacement occurs.

## Exact ABI and scheduling

All values are bf16 bits in contiguous uint16 storage:

| Arena | Layout | Bytes |
|---|---|---:|
| Input | `[batch4, core32, patch2, tile104, IC128]` | 6815744 |
| Weights | `[IC/8, OC/8, innerIC8, innerOC8]`, then 64 scales and biases | 16640 |
| Output | `[batch4, core32, patch2, tile104, OC64]` | 3407872 |

The last batch has 5632 real pixels: 54 complete slots plus 16 pixels in a
partial slot. That slot's remaining rows are zero; the nine unused slots repeat
the batch's slot zero, matching the baseline. Output assembly takes exactly the
first 25600 rows and owns a copy independent of the reusable output BO.

Runtime TAPs preserve core-major/PPC ordering while streaming one patch per
core in each column super-FIFO cycle. Four bounded DMA groups execute serially.
The first group contains eight weight fills, eight input fills and eight output
drains; each later group has eight inputs and eight outputs. Every group's
eight output completions precede all frees and subsequent configurations.
Weight DMA descriptors are freed after first-group completion, while worker
FIFO objects stay acquired until the final batch has completed.

Compiled artifact `/tmp/mdv6-whole-gemm.tAbF2I/whole_gemm.{xclbin,bin}`:

- 72 DMA configurations, 32 output awaits and 40 fill frees; at most one group
  is outstanding, not all 72 descriptors simultaneously.
- All 40 tiles' allocations nonoverlapping and within bounds. L1 input
  `[8192,34816)`, weights `[34816,51456)`, output `[51456,64768)`, RTP and
  barrier ending at 64812 bytes. The 8192-byte stack reserve is below the input.
  This is a tight fit: only 724 bytes remain at the top of the 64 KiB bank map.
- Memtile input `[0,106496)` and output `[106496,159744)`.

The resource claim is for this exact compiled shape, not a generalized memory
estimate. Recheck the map before adding buffering, metadata, or another stage.

## Host ownership and opt-in integration

`whole_gemm.py` owns persistent named input, weight and output arenas plus one
runtime handle. It packs/uploads each input and weights once, submits once,
then explicitly downloads/materializes the host result. It rejects incorrect
shapes/dtypes and invalidates after upload, execution or readback failure; no
retry/fallback is attempted. Weight packing is per call, avoiding unsafe
identity-keyed caching of mutable or short-lived arrays.

Set `MDV6_WHOLE_GEMM_DIR` before starting the process to opt in. Routing is
restricted to `gemm_elan_c4` at 160x160 with IC128/OC64. The same name is used
for an 80x80 RN merge, but its actual IC is 64; it retains normal dispatch and
uses a different artifact. A suspected extra-context problem was therefore not
observed: full runs have 32 cold context-cache misses, zero warm misses and
zero software evictions. These are software observations, not driver traces.

## Validation and transfer accounting

The isolated operator test alternates baseline/experimental execution across
two retained weight variants and four input cases: zero, constant negative,
image/batch-edge impulses, and random. One weight set is the unchanged trained
ELAN2 Conv4; the other deliberately changes convolution rows and BN values to
test rebinding. Sixteen cases pass bitwise exact comparison with finite outputs.
Baseline automatic retry is disabled for this test. Gates remain active under
`python3 -O`; test-only patch setup is outside the operator timer.

| Warm operator API observations | Baseline | Batched GEMM |
|---|---:|---:|
| Submissions / successful completions | 4 | 1 |
| Input/weight upload calls | 5 | 2 |
| Uploaded bytes | 6832384 | 6832384 |
| Download calls | 4 | 1 |
| Downloaded bytes | 3407872 | 3407872 |

Host bytes are unchanged: the baseline already uploads weights only once per
operator and reads all four outputs. What changes is call count, scheduling,
and weight reuse on device. From the generated DMA schedule, eight column
weight fills now occur once instead of four times, reducing logical external
weight reads from 532480 to 133120 bytes per operator. This 399360-byte saving
is schedule-derived, not a measured hardware bandwidth counter. Input/output
DMA volume remains unchanged, and instruction traffic is not included.

Full-model GEMM-only and combined convolution+GEMM routes each pass three then
thirty changing-input frames (seeds 42–71 for each 30-frame run), finite outputs,
and class/vector thresholds 0.5/0.1. GEMM-only uses **450 submissions** instead
of 453; combined uses **414** instead of the convolution-only route's 417.
Every counted runtime call completed. Both 30-frame runs' maximum class/vector
differences are **0.238037109375 / 0.03125**. Warm software context misses and
evictions remain zero; cold context-cache misses are 32.

### Matched inference-only measurements

Separate serial processes, seeds 42–48, one cold frame excluded, six warm
samples per route. Model construction, CPU reference, comparison and reporting
are outside the timer; host packing, CPU islands and detection remain inside.
These numbers are not directly comparable with the historical `--profile`
wall metric. No compiler or other MDV6 hardware job overlapped these runs.

| Route | Warm mean, ms | Submissions | Upload calls |
|---|---:|---:|---:|
| Default | 1589.6 | 453 | 810 |
| GEMM only | 1559.7 | 450 | 807 |
| Convolution only | 1532.1 | 417 | 738 |
| Combined | 1542.1 | 414 | 735 |

GEMM-only is approximately 1.9% faster than default in this short comparison.
But combined is approximately 10 ms (0.65%) slower than convolution-only:
**an additive latency benefit is not established**. Fixed measurement order
(table order), short samples and host/device timing variability limit inference;
do not promote a route or discard the capability on this small difference.
The separate 30-frame warm means are 1561.1 ms GEMM-only and 1536.2 ms combined.
Further work should use longer counterbalanced trials and on-device traces,
then optimize the resident schedule rather than merely count filenames.

GEMM-only warm host bytes remain 424077184 uploaded / 87130112 downloaded;
combined uses 409331584 / 87130112, the same bytes as convolution-only. Reduced
calls do not imply reduced activation traffic between model operators.

The 100/300-frame stages, real-image detection quality and indefinite-memory
behavior have not been validated for these routes. No deliberate timeout or
driver recovery was attempted. Do not promote the default based only on these
short stability/performance measurements.

## Reproduce

Use [README.md](README.md)'s environment and baseline artifacts. Trained
weights SHA-256 remains
`4d0af6e3d80bbbdbcc22a5ec6cf997f66e3f6062e463a04c1da34cf08f44c3ae`.
The installed runtime is under
`/home/jfifield/npu-dev-mdv6/install/mlir-aie/python/aie/utils/hostruntime/`.

```bash
gemm_build=$(mktemp -d /tmp/mdv6-whole-gemm.XXXXXX)
make -C "$gemm_build" -f "$PWD/gemm_conv1x1/Makefile.whole_gemm"
python3 -O gemm_conv1x1/test_whole_gemm.py --build-dir "$gemm_build" --frames 16

# Separate fresh processes; choose unused JSONL report paths.
env -u MDV6_WHOLE_GEMM_DIR -u MDV6_WHOLE_CONV_DIR \
    python3 benchmark_executor.py --frames 7 --report /tmp/mdv6-default-new.jsonl
env -u MDV6_WHOLE_CONV_DIR MDV6_WHOLE_GEMM_DIR="$gemm_build" \
    python3 benchmark_executor.py --frames 30 \
    --report /tmp/mdv6-gemm-new.jsonl
# To combine, also set MDV6_WHOLE_CONV_DIR to the separately built convolution
# artifact directory from FUSION_M1_VALIDATION.md before launching a new process.
```

`run_whole_gemm.lit` builds both this artifact and the unchanged baseline
`gemm_t104_ic128_oc64_p2` comparator, then runs the sixteen-case hardware test.
The expanded host gate includes layout, partial-slot, alias isolation, arena
reuse and failure-invalidation tests. All **79 CPU tests** and the configured
host and fresh-build GEMM hardware lit gates pass. Experimental artifacts and temporary
JSONL/log evidence live under `/tmp/mdv6-whole-gemm.tAbF2I`; the document is
the durable summary and those temporary files may disappear.

Tracking: `mlir-aie-2vb.4` completes this first non-K GEMM increment.
Next (`mlir-aie-2vb.5`): extend bounded spatial batching to a K-blocked production GEMM without
changing its partial-sum rounding, and continue the full-shape SPPELAN gather
proof. Neither successful command batching nor shared artifacts complete the
device-resident inter-operator fusion roadmap.

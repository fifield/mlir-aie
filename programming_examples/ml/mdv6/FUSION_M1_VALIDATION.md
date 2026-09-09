# Whole-convolution sequencing — 2026-09-09

First Milestone 1 operator implemented, experimentally integrated, and tested.
The default path remains unchanged. This builds on
[the persistent baseline](FUSION_M0_VALIDATION.md) and advances
[the fusion roadmap](FUSION_PERF_PLAN.md); it is command batching, **not**
inter-operator on-chip fusion. Work started from `d497d6b3b`.

## What changed

The `mc_re8_rn3` Conv2 inside re8/re21 bottlenecks has logical input/output
`20x20x64`, kernel 3x3, stride one, padding one. The original route uses 32 cores,
8x8 output tiles, 16-channel output blocks, one patch per core. Nine spatial
tiles fit in one padded 32-core batch, but four output-channel blocks require
four host submissions. The new coherent program executes all four blocks
within one submission, preserving that geometry and the existing kernel.

This operator occurs twelve times per model frame. Replacing four calls by
one removes 36 submissions: **453 -> 417**. It does not implement the general
spatial multi-batch case or GEMM batching yet.

`conv/aie2_multicore.py` gains optional `output_blocks=1`; the experimental
wrapper fixes it to four. A compact worker loop consumes four distinct weight
blocks. The runtime writes RTP once, sets one invocation barrier per worker,
and schedules four sequential DMA task groups. Every group waits for all eight
column outputs before freeing fills or starting the next group. Workers release
the invocation barrier only after all four blocks. No `.bin` concatenation.

Default generator output for the selected shape was checked byte-for-byte
against the previous committed generator and is identical. The new wrapper and
Makefile use separate artifacts; baseline xclbins were not overwritten.

## Physical contracts and ownership

All external values are bf16 bit patterns transported as uint16:

| Arena | Physical layout | Bytes |
|---|---|---:|
| Input | `[core=32, patch_h=10, patch_w=10, IC=64]` | 409600 |
| Weights | `[OCblock=4, packed_weights_and_BN=9248]` | 73984 |
| Output | `[OCblock=4, core=32, tile_h=8, tile_w=8, OC=16]` | 262144 |

Real spatial patches are row-major. Unused cores repeat patch zero, matching
the current baseline contract. Patch halos outside the image are zero. Final
assembly crops partial right/bottom tiles to the logical 20x20 output.
Each weight slot is packed `[OC/8, IC/8, kernel9, innerIC8, innerOC8]` followed
by sixteen BN scales and sixteen biases.

`whole_conv.py` owns one persistent named arena of each kind and a runtime
handle. It explicitly packs host patches, uploads input/weights, submits once,
downloads the output, and reconstructs a host HWC tensor. Weights are repacked
per operator use in this first implementation; there is no id-keyed cache or
assumed cross-frame on-chip residency. Failure invalidates this executor and
does not trigger a retry/fallback; recover hardware as needed and restart.

`MDV6_WHOLE_CONV_DIR` enables only the exact `mc_re8_rn3` shape and is read at
module import. It routes before baseline variant probing, so it does not load
an unused per-shape context. Wrong shapes fail before dispatch; missing
experimental artifacts fail explicitly. Unset the variable and start a fresh
process to restore the baseline. Other layers and saved baselines are unchanged.

## Compiled resource/synchronization proof

Fresh artifact: `/tmp/mdv6-whole-conv.CVzujK/whole_conv.{xclbin,bin}`.
The corresponding `whole_conv.mlir.prj/input_with_addresses.mlir` contains:

- Four groups of 24 DMA descriptors: eight weight fills, eight input fills,
  eight output drains. 96 configurations total; not 96 simultaneously live.
- 32 output awaits total. Each group's eight output waits precede its input/
  weight frees and all subsequent group configurations.
- 192 RTP scalar writes and 32 barrier sets once per invocation.
- All 40 tiles' buffers nonoverlapping/in bounds in the reviewed map. Maximum
  L1 allocation end 37484 bytes; memtile input 51200 bytes at address zero and
  output 8192 bytes at 65536, ending at 73728 bytes.

Bank-aware allocation warned and fell back to successful sequential placement.
The artifact compiled and repeated execution passed; estimates alone were not
used as the resource gate. These observations apply to this exact shape/build.

The device still rereads the 409600-byte input arena for every OC block.
Host upload savings are **not** device-DMA savings. Holding patches on-chip
across OC blocks is a future optimization and must budget their lifetime.

## Numerical and capability gates

The initial twelve-case operator run passed bitwise comparison against the
existing kernel using all twelve trained Conv2 weight sets. Review then expanded
the test to the full Cartesian set: all twelve weights under zero, constant
negative, corner impulses, and random inputs (48 cases). The fresh-build
`run_whole_conv.lit` passes this stronger test under `python3 -O`, building
both the experimental artifact and its baseline comparator. Gates use explicit
exceptions, not disabled assertions. This is equivalence to the deployed
computation, not a new detection-quality evaluation.

Warm isolated-operator counters:

| API observation | Baseline | Whole operator |
|---|---:|---:|
| Submissions / successful returns | 4 | 1 |
| Upload calls | 8 | 2 |
| Uploaded bytes | 1712384 | 483584 |
| Download calls | 4 | 1 |
| Downloaded bytes | 262144 | 262144 |

The complete model passes 3 then 30 changing-input frames with strict finite
outputs and class/vector thresholds 0.5/0.1. Thirty-frame maximum differences:
**0.238037109375 / 0.03125**; every frame completes exactly 417 submissions.
Warm full-frame uploads become 738 calls / 409331584 bytes versus baseline
810 / 424077184. Downloads become 417 calls with unchanged 87130112 bytes.
Twenty runtime load calls remain warm cache hits; no software eviction observed.

Runtime counters are the actual tensor API calls/whole-BO sizes, not device
bandwidth traces. CPU reference, model construction and reporting remain outside
the inference-only benchmark timer; host layout work/CPU islands stay inside.
Do not compare this timer directly with historical `--profile` wall time.

Matched seven-frame runs, separate serial processes with the same seeds 42–48
and one cold frame excluded, measured **1599.4 ms default / 1554.4 ms whole**
over six warm frames: 45.0 ms lower, approximately 2.8%, in this short comparison.
Both runs passed strict numerical gates. No compiler or other MDV6 job ran
concurrently. Order was default then experimental, not randomized trials;
do not treat this as a precisely established long-term speedup. The separate
30-frame experimental run averaged 1563.5 ms over its 29 warm frames.

The 100/300-frame stages, indefinite-memory proof and real-image quality tests
have not run for this route. No timeout or recovery test was attempted. The
experimental route is not promoted to default on these short-stage results.

## Reproduce

Use the environment, trained weights and baseline build root in
[README.md](README.md). Baseline artifacts for the full model must already
exist. The isolated operator comparator requires `mc_re8_rn3` specifically.

```bash
whole_build=$(mktemp -d /tmp/mdv6-whole-conv.XXXXXX)
make -C "$whole_build" -f "$PWD/conv/Makefile.whole_conv"
python3 -O conv/test_whole_conv.py --build-dir "$whole_build" --frames 48

# Separate processes; unique report paths preserve previous evidence.
env -u MDV6_WHOLE_CONV_DIR -u MDV6_WHOLE_GEMM_DIR python3 benchmark_executor.py --frames 7 \
    --report /tmp/mdv6-default-new.jsonl
MDV6_WHOLE_CONV_DIR="$whole_build" python3 benchmark_executor.py --frames 7 \
    --report /tmp/mdv6-whole-new.jsonl
MDV6_WHOLE_CONV_DIR="$whole_build" python3 benchmark_executor.py --frames 30 \
    --report /tmp/mdv6-whole-stream-new.jsonl
```

Full-model weights SHA-256 and installed runtime are unchanged from M0.
Logs/JSONL are under `/tmp/mdv6-whole-conv.CVzujK`: `operator-test.jsonl`,
`full-{3,30}.{log,jsonl}`, `default-7.{log,jsonl}`, `whole-7.{log,jsonl}`, and
`lit.log`. These are temporary evidence; this document is the durable summary.
CPU suite: 66 passing tests, including opt-in isolation, shape failures,
weight packing, halo/crop ordering and SPP schedule checks.
Configured `run_host_checks.lit` and fresh-build `run_whole_conv.lit` pass.

## Next unproven capabilities

`mlir-aie-2vb.4`: apply coherent whole-operator sequencing to one GEMM while
preserving its OC/K blocking and bf16 rounding; then generalize to additional
convolution/spatial batches. Full-model internal activations still return to
host. These dispatch savings are an enabling step toward resident regions.

`mlir-aie-2vb.2`: the new [SPP9 schedule](sppelan/FUSION_SCHEDULE.md) proposes
16 workers, full-spatial eight-channel pooling shards, and 16-pixel gather
stripes. Its checked phase budgets peak at 58432 bytes/core and 221440
bytes/memtile. **Physical phase-buffer aliasing and cross-column routing are
not proven.** Next compile a synthetic full-size gather-only sentinel test;
do not treat the arithmetic budget as executable placement or close the issue.

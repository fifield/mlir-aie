# SPP9 full-spatial projection/pooling shard

Date: 2026-09-09. Tracking: `mlir-aie-2vb.2.2`; implementation follows the
[gather handoff](GATHER_VALIDATION.md#next-bounded-implementation), based on
commit `5fdbbcef0`. The parent first-island task remains incomplete.

## Capability and scope

One compute worker now projects all 400 spatial positions to eight channels,
then performs three local 5x5 max-pools. It holds the projection weights
through all 25 input stripes and retains all four feature planes in one L1
output allocation. There is one host submission and no host synchronization
between projection stripes or pooling stages. Shared production kernels and
the default full-model route are unchanged.

This is an arithmetic/residency proof for **one of sixteen channel shards**,
not the complete SPP9 operator. Diagnostic output still exports all four
planes. The proof is not yet connected to the device gather or final projection;
physical phase-A/B aliasing and a combined worker-channel schedule remain.
It does not establish a full-model speedup or cross-frame weight residency.

## Frozen interface and numerical contract

All BOs carry uint16 storage for bf16 bits, except the raw metadata words.

| Program argument | Shape | Bytes |
| --- | --- | ---: |
| Shard input I | `[400,256]` | 204800 |
| Shard packed weights W | `[2,1056]` | 4224 |
| Four output planes O | `[level4,400,8]` | 25600 |
| Metadata M | `[32]` | 64 |
| Pool-only input I | `[400,8]` | 6400 |

The normal argument list is `[I,W,O,M]`; pool-only uses `[I,O,M]`. Metadata
word 0 observes `get_rnd()` without changing the rounding mode; the remaining
31 words are reserved zeros. The host requires all words zero, including
observed floor mode 0. Both O and M are awaited before input tasks are freed.

[phase_a_shard.cc](phase_a_shard.cc) reuses the corrected production
`gemm_conv1x1_kblocked_bf16` and literal-row extraction helper. SPP conv1's
deployed full-width configuration is `gemm_t28_ic256_oc128_kb128_p1`.
The shard preserves its **two ordered K128 calls**, bf16 intermediate partial
store, and last-K-only fused BN/SiLU. It does not substitute an uninterrupted
K256 reduction or the older scalar SPP convolution approximation.

For each eight-channel slice, one weight chunk consists of 1024 blocked matrix
values, eight BN scales, eight biases, and sixteen zero padding elements.
The resulting 1056-element physical stride keeps both chunks 64-byte aligned;
the 1040-element logical size alone would misalign the second chunk. CPU tests
round-trip every matrix/BN slice and check alignment/padding.

Input stripes contain 16 consecutive pixels, each with all 256 input channels.
The worker acquires the two weight chunks and one full output object, consumes
25 input FIFO objects, writes f0 directly into the output plane, then pools
only after f0 is complete. There is no duplicate 25-KiB feature/readback buffer.
The pool-only artifact copies a diagnostic f0 into that same output allocation
before applying the identical pooling function.

Pooling uses valid row-major window traversal, starting at negative infinity.
Strict greater-than retains the first equal maximum, including signed-zero
ties. The independent host oracle uses padded sliding windows and first-maximum
selection; CPU tests compare it bitwise with Torch max-pooling. The diagnostic
contract requires finite bf16 inputs/outputs; NaN/Inf propagation is not tested.

## Compiled placement gate

[check_phase_a_ir.py](check_phase_a_ir.py) checks the actual compiled map and
ELF allocation sections for both artifacts. Each program has exactly one
compute core, no neighbor-core memory borrowing, and one full feature buffer.

| Allocation | Shard byte interval | Pool-only byte interval |
| --- | --- | --- |
| Reserved stack | `[0,8192)` | `[0,8192)` |
| Four feature planes | `[8192,33792)` | `[8192,33792)` |
| One input stripe / f0 | `[33792,41984)` | `[33792,40192)` |
| Two packed weight chunks | `[41984,46208)` | absent |
| Metadata | `[46208,46272)` | `[40192,40256)` |
| Compiler synchronization state | `[46272,46288)` | `[40256,40268)` |

Shard text is 15600 bytes; pool-only text is 2048 bytes. Both ELFs have zero
allocated data/BSS outside the map. The generated linker script declares
128-KiB program memory; do not infer a 16-KiB instruction limit from historical
builder comments. The map proves reserved stack separation, not a measured
runtime stack high-water mark. This single-phase allocation does not prove
the proposed shared phase-A/B arena or sixteen-worker combined placement.

The checker fails on unknown buffer layouts/types, overlapping allocations,
hidden feature copies, neighbor borrowing, missing metadata completion, or
unaccounted ELF data/BSS. It is a focused textual-IR checker, not a general
MLIR/ELF verifier or independent physical-route decoder.

## Hardware evidence

Root ran hardware serially after independent kernel, host, and compiled-map
reviews. The frozen generator/kernel were built in `/tmp/mdv6-phase-a.YCX1BL`;
the corrected full-width baseline was freshly built into that same root's
`gemm/` directory. Platform: Strix Halo NPU2, kernel `6.17.0-20-generic`,
installed mlir-aie under `/home/jfifield/npu-dev-mdv6/install/mlir-aie`.

| Gate | Result | Warm shard host wall, first shard frame excluded |
| --- | --- | ---: |
| Independent pool-only boundary gate | 17 cases PASS | not a shard measurement |
| Projection smoke, all 16 trained slices | 16 frames PASS | 9.341 ms |
| All 16 slices × five input categories | 80 frames PASS | 9.311 ms |
| Sustained changing-input/slice run | 100 frames PASS | 9.401 ms |
| Sustained changing-input/slice run | 300 frames PASS | 9.350 ms |

Every projection run also repeats the 17 independent pool cases. The five
input categories are random, constant negative, stripe/baseline-boundary
impulses, weighted near-cancellation across the K128 boundary, and zero.
These are constructed inputs with trained SPP weights, not captured full-model
activations. The cancellation input is bf16-quantized and is not claimed to
produce an exact zero. Frame-to-frame channel slices change the loaded weights.

All 3200 f0 values per shard match the corresponding eight channels of the
fresh full-width projection **bitwise**. All three subsequent planes match
independent CPU pooling of the observed f0, also bitwise; every one of 12800
output values is checked. The independent pool tests cover negative constant
and border values, all four corners, all eight channels, random values, zero,
and alternating signed-zero ties. They validate pooling separately from any
shared convolution arithmetic. Required rounding metadata is floor mode 0 in
every observed frame; no mode is set by the proof.

Each shard run records one successful submission, two uploads (209024 bytes),
and two downloads (25664 bytes including diagnostic metadata). Pool-only uses
one upload (6400 bytes), one submission and the same two downloads. There are
no intermediate host transfers. The measured frame scope records no loads,
context cache misses, or evictions. Initial loads/allocations, weight packing,
baseline execution, and CPU reference construction/comparison are outside the
shard timer. Host copy, sync, submit/wait, readback/copy and output-validity
checks are inside it. Baseline and shard contexts are both used by the harness;
driver context switches and device cycles/traffic remain unmeasured.

The pool-only means in these runs are approximately 8.06–8.46 ms, indicating
that the initial scalar pooling is an optimization candidate. These are
diagnostic host timings, **not** a matched baseline speedup, a sixteen-worker
latency prediction, or proof of a full-model performance change.

Temporary logs and build evidence:

- `/tmp/mdv6-phase-a.YCX1BL/pool-first.jsonl`
- `/tmp/mdv6-phase-a.YCX1BL/projection-{16,80,100,300}.jsonl`
- `/tmp/mdv6-phase-a.YCX1BL/resource-gate.json` and `baseline-build.log`
- Generator SHA256: `caa30d387ae25d18cff0428811ce22eca836ed65f35449f73d6e30aca8d9a479`.
- Kernel SHA256: `6b9b501008d4dbb77a397f9a6958c77449441081d4eac4a250e96553400ffc7b`.
- Trained weights SHA256: `4d0af6e3d80bbbdbcc22a5ec6cf997f66e3f6062e463a04c1da34cf08f44c3ae`.
- Shard xclbin / instructions SHA256:
  `7006ada2d15cd82521cfd48a7fb9e5df2a90333c3ca0e043a2120d7cc8c3a835` /
  `87c34cb5f05bba6fbbe97b0e46e72d3e9de0b96d035fa02b483d21f14c2b1b95`.
- Pool xclbin / instructions SHA256:
  `cbb786d04b7f2c8ea75e8276c2bc57e42e8203caacb85e7ff5a0dd880c97a374` /
  `60c286c7ab641f442713897cac9d38ae162c1c6f261785bdd42e18316c03e625`.

Temporary artifacts may disappear; the commands below reproduce the gate.

## Reproduce in a fresh context

```bash
source /home/jfifield/npu-dev-mdv6/env.sh
export PYTHONPATH="/home/jfifield/npu-dev-mdv6/install/mlir-aie/python${PYTHONPATH:+:$PYTHONPATH}"
cd /home/jfifield/npu-dev-mdv6/mlir-aie/programming_examples/ml/mdv6
phase_a_build=$(mktemp -d /tmp/mdv6-phase-a.XXXXXX)
make -C "$phase_a_build" -f "$PWD/sppelan/Makefile.phase_a_shard"
python3 -O sppelan/check_phase_a_ir.py --build-dir "$phase_a_build"
MDV6_BUILD_DIR="$phase_a_build" python3 gemm_conv1x1/build_gemm_conv1x1.py \
    gemm_t28_ic256_oc128_kb128_p1
PYTHONDONTWRITEBYTECODE=1 python3 -O -m unittest \
    sppelan.test_phase_a_host sppelan.test_phase_a_ir
python3 -O sppelan/test_phase_a_shard.py --build-dir "$phase_a_build" \
    --pool-only --failure-dir "$phase_a_build/failures-pool"
python3 -O sppelan/test_phase_a_shard.py --build-dir "$phase_a_build" \
    --baseline-dir "$phase_a_build" --frames 80 \
    --failure-dir "$phase_a_build/failures-80"
```

The trained `mdv6_bf16_weights.pt` file must be available as described in
[README](../README.md). Never accept random initialized weights as a reference.
Build the named baseline freshly: updating Python does not repair stale device
artifacts. Each Makefile target declares both xclbin and instruction outputs.

Run NPU jobs serially with sources/artifacts frozen. After success, repeat with
100 then 300 frames and distinct failure directories. The minimum 16-frame
projection smoke covers every trained shard; the default 80 frames cover all
16 shards across all five input categories. The runner disables the historical
baseline reload/retry path. Any runtime, exact-result, metadata, or nonfinite
failure stops and invalidates further use. Optional exclusive NPZ evidence
includes inputs, weights, expected/observed outputs when available, and error;
baseline failures also preserve their full input/weights. Recover a timed-out
device before further hardware work, never by blind retry.

[run_spp_phase_a_shard.lit](../run_spp_phase_a_shard.lit) builds both diagnostic
artifacts and the selected full-width baseline freshly, checks the map, and
runs the 80-frame matrix. Use configured `lit -j1` for hardware tests. CPU tests
are included in [run_host_checks.lit](../run_host_checks.lit).
The configured CPU, fresh shard/reference build-and-run, and gather regression
tests all passed (3/3). The complete CPU suite also passes 127 tests under
`python -O`, including eleven new host tests and five resource-checker tests.

## Next integration cut

Keep the independent gather and arithmetic proofs as regression targets.
`mlir-aie-2vb.2.3` next needs a **one-column, four-worker packet-aggregation
sentinel proof** before sixteen-worker expansion, followed by explicit L1
phase-alias ownership. A candidate combined memtile budget is:

| Channels | Proposed use |
| --- | --- |
| S2MM 0–3 | Existing cross-column gather receivers |
| S2MM 4 | Four packet-merged worker outputs: phase-A features, later phase-B output stripes |
| S2MM 5 | Combined host ingress for weights and phase-A input stripes |
| MM2S `column` | Existing retained-feature multicast |
| MM2S 4 | Phase-A input / phase-B gathered-stripe multicast to workers |
| MM2S 5 | Addressed worker weights and small output-grant tokens |
| MM2S `(column+1)%4` | Final output to shim |

This uses all six receive and four transmit channels per memtile. It is a
**proposed schedule**, not compiled proof that the combined routes, descriptors,
locks, switch ports, or worker memory fit. Packet IDs do not make a shared
receive DMA choose a destination address by sender. Issue explicit sequential
grants: worker 0 sends its 25600-byte planes into row 0, receive completion
unlocks worker 1, then workers 2 and 3. Only one worker may send at a time.
Later phase-B output joins can use the same receive channel with strided
placement and the same grant discipline. Reusing descriptors alone does not
resolve stream-route conflicts or stale queued DMA at phase transitions.

API precedent: [the placed packet-switch example](../../../basic/packet_switch/aie_add_placed.py)
merges compute-tile packet producers into one memtile DMA using `packetflow`,
`dma_bd(..., packet=(type,id))`, explicit buffers and locks. It uses low-level
DMA because ObjectFifo packet-flow support is unavailable in that example.
This establishes a local API mechanism, not feasibility of this larger schedule.

First compile one column/four workers with the channel numbers reserved above,
four sequential grants, and a worker-major 100-KiB destination. Check descriptor
chains, lock lifetimes, packet routes and physical allocations before sentinel
hardware tests. Then add phase-B strided output joins and four-column gather
coexistence. Separately prove the shared L1 arena from the gather handoff:
all phase-A feature stores must reach L2 before any overlapping phase-B write.
Finally connect arithmetic and gather into full SPP9. Scalar pooling can be
vectorized after correctness/residency is established; it is not yet optimized.
That follow-up is tracked separately as `mlir-aie-2vb.8`; preserve the exact
finite-value and signed-zero regression gates when optimizing it.

# MDV6 Implementation Architecture Status

State as of Phase G (commit b09af7538). Wall 1265 ms warm, 190 launches,
0.79 fps, max_class_diff=0.1506, max_vector_diff=0.0000 (bytewise correct).

## ELF Build Formats (lowest → highest fusion)

### 1. Standalone .xclbin + .bin (legacy path)
- Built by `build_multicore.py` / `build_gemm_conv1x1.py`.
- One layer per file pair, loaded via `DefaultNPURuntime.load(NPUKernel(xclbin, insts))`.
- Each loaded xclbin burns one of 32 XRT context slots → fragmented.
- Still wired as the fall-through path when no merged ELF exists.

### 2. Single-clone merged ELF — Phase A
- Built by `build_x1_mc.py` / `build_x1_gemm.py`.
- Wraps the standalone `aie2_multicore.py` output in a 1-sub merged dispatcher.
- One ELF with `@main` entry point, loaded via `xrt.elf` + `xrt.ext.kernel(ctx, "main")`.
- No xclbin context burn — fixed the fragmentation problem from #1.
- Same dispatch count as standalone, just a different loading mechanism.
- Examples: `merged_re8_rn3_x1`, `merged_aconv7_x1`, etc. (the bulk of legacy layers).

### 3. Multi-clone batch fanout — Phase A.1
- `build_merged.py` with N stock sub-devices, all the same kernel.
- N `aiex.configure / aiex.run` pairs in the dispatcher's `@main`, sequential.
- All N sub-runs share the array (32 cores) — they take turns, not parallel.
- Each sub handles a different input/output via separate dispatcher args (no chain).
- **Saves N-1 launch_gap per layer-call. NPU dispatch overhead unchanged** (each sub-run still pays it).
- Examples: `merged_ftconv0_x8`, `merged_aconv3_x4`, `merged_aconv16_x4`, `merged_elan_c3_p4_x4`.

### 4. Pair merged ELF with chain_links — Phase C step A
- Two sub-devices with **different weights, shared input** via `chain_links=[(0,0,1,0)]`.
- Dispatcher signature after aliasing: `(W_a, in_shared, out_a, W_b, out_b)` — input collapsed to one BO.
- Used by `run_gemm_pair_mc` for the RepNCSP `conv1+conv2` GEMM pair (1×1 convs on the same input).
- Measured: -45 ms wall (incl. -7 ms NPU — dispatch-overhead amortization across the two sub-runs).
- Examples: `merged_gemm_re4_rn1_pair_x1`, `merged_gemm_re6_rn1_pair_x1`, `merged_gemm_re8_rn1_pair_x1`.

### 5. OCB-unrolled ELF — Phase E/F/G (the big mechanism)
- Built by `build_ocb.py` → `aie2_multicore_ocb.py` (forked from `aie2_multicore.py`).
- Single sub-device, but the **OC-block loop is unrolled inside the runtime sequence at compile time**.
- 4D-strided memtile TAP (`sizes=[n_ocb, 1, 1, slot_size]`) iterates `n_ocb` times in hardware.
- Effective ppc absorbs spatial batching via the existing `patches_per_core` compile-time unroll.
- One xrt.run replaces `(n_ocb × n_spatial_batches)` host-level dispatches per layer-call.
- Examples: `ocb_re6_rn3_x1`, `ocb_re8_rn3_x1`, `ocb_re6_c3_x1`, `ocb_re8_c3_x1`, `ocb_re4_c3_x1`,
  `ocb_aconv3_x1`, `ocb_aconv7_x1`, `ocb_aconv16_x1`, `ocb_aconv19_x1`.

## aie.device Stitching Mechanism

The mlir-aie compiler produces ONE binary containing N independent device blocks plus a dispatcher device:

```mlir
module {
  aie.device(npu2) @sub0_mc_re6_rn3 {
    // compute tiles, memtiles, shim, ObjectFifos, workers
    aie.runtime_sequence @sub0_mc_re6_rn3_seq(%in, %wt, %out) { ... }
  }
  aie.device(npu2) @sub1_... { ... }
  aie.device(npu2) @main_dispatcher {
    aie.runtime_sequence @main(%arg0, %arg1, %arg2, ...) {
      aiex.configure @sub0_mc_re6_rn3 {
        aiex.run @sub0_mc_re6_rn3_seq(%arg1, %arg0, %arg2)
      }
      // more aiex.configure/aiex.run pairs for multi-clone or pair fanout
    }
  }
}
```

Key behaviors:
- Each `aiex.configure { aiex.run ... }` is one sub-run inside the xrt.run.
- BD slots reset between sub-devices (each sub starts with fresh shim/memtile BD allocation).
- `share_arg_idxs={1}` in `build_merged.py` promotes arg-1 of each sub (weight by convention) to dispatcher arg-0, then per-sub `(in, out)` args follow.
- `chain_links=[(src_sub, src_arg, dst_sub, dst_arg)]` aliases one sub's arg to another's — used for shared input on the pair pattern.
- Sub-runs within one xrt.run are **sequential, not parallel** on the array.

## Kernel Fusion (PythoC .o files in `kernels/build/`)

Three kernel object files do all the actual compute:

| Kernel | Op fusion | Used by |
|---|---|---|
| `conv3x3_fused_packed_bf16` | Conv3×3 + BN + SiLU (packed weight layout for mmul<4,8,8>) | All mc_* and ocb_* 3×3 layers |
| `gemm_conv1x1_fused_packed_bf16` | GEMM (1×1) + BN + SiLU | All gemm_* layers without K-blocking |
| `gemm_conv1x1_kblocked_bf16` | GEMM (1×1) with K-blocking (split IC across kernel calls) | Large 1×1 layers where full IC doesn't fit L1 |

The "fused" part is **kernel-level**: Conv+BN+SiLU happens in one C function call per tile, with no host-side intermediate between conv output and BN/SiLU.

### Phase D: weight-level fusion

`fuse_repconv` (in `_full_model_helpers/elan_test_tiled.py`) collapses RepConv's two parallel BN+conv branches:

```
SiLU(BN(Conv3x3(x)) + BN(Conv1x1(x)))   →   SiLU(Conv3x3_fused(x))
```

The 1×1 weights are added to the 3×3 kernel's center position; both BN params are folded in. The result drops into the existing `conv3x3_fused_packed_bf16` kernel with `bn_w=1`. This moved RepConv from CPU (~600 µs/iter) to NPU.

## Dispatch Path (in `run_tiled_mc.py`)

```
run_tiled_fused_conv_mc(mc_name, ...)
  → _run_tiled_fused_conv_mc_impl(...)
    actual_name, ppc = _get_mc_variant(mc_name)         # picks _p2/_p4 if exists

    1. if actual_name in _MERGED_LAYERS_OCB:
         → _run_tiled_mc_inner_ocb_merged(...)          # Phase E/F/G path
    2. elif actual_name in _MERGED_LAYERS:
         → _run_tiled_mc_inner_merged(...)              # Phase A/A.1 path
    3. else (regime-active xclbin):
         → _run_tiled_mc_inner(mc_kh, ...)              # legacy xclbin path
```

GEMM has parallel structure:
- `run_gemm_conv1x1_mc` → `_run_gemm_oc_blocked_merged` (single-sub) | `_run_gemm_kblocked_merged` (K-blocked)
- `run_gemm_pair_mc` → `_run_gemm_oc_blocked_pair_merged` (chain_link pair)

## Registries (which layers use which mechanism)

```python
# Phase E/F/G OCB-unroll
_MERGED_LAYERS_OCB = {
    # Phase E (3×3 rn3):
    "mc_re8_rn3":      ("ocb_re8_rn3_x1",  n_ocb=4, ppc=1,  oc_block=16),
    "mc_re6_rn3":      ("ocb_re6_rn3_x1",  n_ocb=3, ppc=4,  oc_block=16),
    # re4_rn3 NOT in OCB — legacy already at 1 dispatch/layer-call.

    # Phase F (3×3 c3):
    "mc_re8_c3":       ("ocb_re8_c3_x1",   n_ocb=8, ppc=1,  oc_block=16),
    "mc_re6_c3":       ("ocb_re6_c3_x1",   n_ocb=6, ppc=4,  oc_block=16),
    "mc_re4_c3_p2":    ("ocb_re4_c3_x1",   n_ocb=4, ppc=16, oc_block=16),

    # Phase G (3×3 stride-2 aconv):
    "mc_aconv3":       ("ocb_aconv3_x1",   n_ocb=8,  ppc=4,  oc_block=16),
    "mc_aconv7":       ("ocb_aconv7_x1",   n_ocb=16, ppc=1,  oc_block=16),
    "mc_aconv16":      ("ocb_aconv16_x1",  n_ocb=6,  ppc=4,  oc_block=16),
    "mc_aconv19":      ("ocb_aconv19_x1",  n_ocb=16, ppc=1,  oc_block=8),
}

# Phase A/A.1 single- and multi-clone merged ELFs
_MERGED_LAYERS = {
    "mc_ftconv0":      ("merged_ftconv0_x8",      8),   # 8-clone fanout
    "mc_ftconv1_p2":   ("merged_ftconv1_p2_x4",   4),
    "mc_elan_c3_p4":   ("merged_elan_c3_p4_x4",   4),
    "mc_aconv3":       ("merged_aconv3_x4",       4),
    "mc_aconv16":      ("merged_aconv16_x4",      4),
    # ... single-clone x1 ELFs for every other 3×3 layer (Phase A) ...
}

# GEMM pair merged ELFs (Phase C step A) — chain_links for shared input
# merged_gemm_re6_rn1_pair_x1, merged_gemm_re4_rn1_pair_x1, merged_gemm_re8_rn1_pair_x1
```

## What's NOT Fused (and Why)

| Layer / Pattern | Why not OCB-unroll |
|---|---|
| mc_re4_rn3 | Caller passes oc_block=32 (full OC) → legacy already 1 dispatch/layer-call → nothing to collapse. Phase E shipped a buggy "win" here that was fixed in Phase G (wrong RTP setup made the kernel do half the work). |
| mc_elan_c3 | Spatial 160×160 → 1600 patches at tile_4 → would need effective_ppc≥50, exceeds L1. Already uses merged_elan_c3_p4_x4 fanout. |
| mc_aconv5 | total OC=192, oc_block=16 → n_ocb=12 × ppc=4 = 48 unrolled iterations → exceeds program memory for stride=2. oc_block=24 fits count but breaks 8-wide SIMD. |
| GEMM | Kernel has no OC-block decomposition (computes full OC per call) — Phase E/F/G mechanism doesn't apply. Pair fusion (Phase C step A) is the main GEMM win. |
| Cross-bottleneck-iter | Data dependency (residual feedback i → i+1). Multi-layer fusion possible but multi-week. |
| Cross-c3-call within run_re_mc | x4 c3 input = function of x3 c3 output. Sequential, no overlap. |

## Hardware/Compile Constraints That Bound Further Work

| Resource | Limit | Where it bites |
|---|---|---|
| L1 per core | 64 KB | Caps ppc, oc_block, and patch sizes |
| Memtile BD count | 48 across 12 channels | Manageable with 4D-strided BDs (Phase F fix) |
| Shim BD count | 16 per channel | Same fix applies |
| **Program memory per core** | **~16 KB** | **Caps total unrolled kernel iterations at ~40 (stride=2) / ~50+ (stride=1)** — hit when building aconv5 |
| L2 per memtile | 512 KB | Plenty for current patterns |
| Compute precision | bf16 (5.12 TFLOPS peak) | Not bfp16 (41 TOPS) — retargeting would be a major redesign |
| Compute utilization | <0.1% of peak | Per-dispatch overhead dominates, not arithmetic |

## Where Wall Time Goes (Phase G state, ~1265 ms warm)

```
├── npu_run           ~902 ms (66%)   NPU active across 190 dispatches
│                                     ~4750 µs avg per dispatch
│                                     of which ~5-50 µs is real compute
│                                     and ~5-25 µs is real DMA
│                                     → ~99% is dispatch setup/sync/descriptor processing
├── launch_gap        ~290 ms (22%)   Python/pyxrt plumbing (~1500 µs/call avg)
├── pre_post          ~145 ms (10%)   Model setup + last layer
├── cpu_layers         ~12 ms          Detection, AvgPool, Upsample (RepConv moved to NPU in Phase D)
└── numpy/fuse misc    ~17 ms          np.concatenate, fuse_bn cache lookups
```

**Per-dispatch NPU-side overhead is the dominant remaining cost.** At ~4750 µs/dispatch with only ~50 µs of useful work each, ~99% is going into dispatch infrastructure (kernel start/stop, lock synchronization, BD descriptor processing inside the NPU controller). This is the fundamental cost to attack to push below ~1100 ms.

## Wall Trajectory Through All Phases

| Stage | Wall | Launches | max_class | max_vec | Notes |
|---|---|---|---|---|---|
| Pre-Phase-A xclbin | 1405 | 453 | 0.219 | 0.000 | baseline |
| + Phase A (ELF-only) | 1375 | 368 | — | — | xclbin → merged-x1 |
| + Phase C step A | 1330 | 354 | — | — | RN1 pair |
| + Phase D | 1640 | 468 | 0.219 | 0.000 | RepConv on NPU (+250 ms intentional) |
| + Phase E (rn3 OCB) | 1396 | 324 | 0.251 | 0.031 | (re4_rn3 had silent bug — fixed in G) |
| + Phase F (c3 OCB + 4D BD) | 1369 | 254 | 0.227 | 0.031 | |
| **+ Phase G (aconv + re4 fix)** | **~1265** | **190** | **0.151** | **0.000** | bytewise correct |

Net change vs Phase D regression: **-375 ms (-23%)**. We are 140 ms below the pre-Phase-A xclbin baseline with RepConv-on-NPU still active and conv outputs bytewise-correct against the torch reference.

## Source Layout

```
mdv6/
├── test_full_model_mc.py            # Top-level driver; defines run_re_mc, run_rn_mc, run_aconv_mc
├── profile_harness.py               # Profiler wrappers (_xrt_run_kernel hook for per-layer attribution)
├── run_tiled_mc.py                  # Dispatch entry + all _run_*_merged variants + registries
├── regime_config.py                 # Active-shape lookup (tile_h/oc_block/stride) per layer
├── run_full_model.lit               # End-to-end CI test (build merged ELFs + run model)
├── Makefile                         # build / profile / clean targets
├── .gitignore                       # Excludes mdv6_bf16_weights.pt + onnx graph + logs
├── lit.local.cfg                    # Sets mdv6_weights feature when weights.pt is present
├── _full_model_helpers/
│   └── elan_test_tiled.py           # extract_patch, fuse_bn, fuse_repconv, etc.
├── conv/
│   ├── aie2_multicore.py            # IRON gen for 3×3 conv (production stock kernel)
│   ├── aie2_multicore_ocb.py        # IRON gen for OCB-unrolled (Phase E/F/G fork)
│   ├── mc_configs.py                # MC conv shape registry
│   ├── gemm_configs.py              # GEMM 1×1 shape registry + sizing helpers
│   ├── build_merged.py              # Dispatcher-stitching builder (the workhorse)
│   ├── build_x1_mc.py               # Builds x1 + multi-clone fanout MC merged ELFs
│   ├── build_x1_gemm.py             # Builds GEMM merged ELFs (kblocked + oc-blocked)
│   ├── build_pair_rn1.py            # Builds GEMM pair ELFs (Phase C step A)
│   ├── build_pair_rn3.py            # Phase H step-1 spike (chain_links bottleneck pair)
│   ├── build_ocb.py                 # Builds OCB-unrolled ELFs (Phase E/F/G)
│   ├── test_pair_rn1.py             # Standalone correctness: GEMM pair (re6_rn1)
│   ├── test_ocb_re6_rn3.py          # Standalone correctness: OCB re6_rn3
│   ├── test_ocb_re8_rn3.py          # Standalone correctness: OCB re8_rn3
│   ├── trace_ftconv0.py             # HW trace runner (diagnostic)
│   └── build_merged/                # ELF + MLIR + per-build artifacts (gitignored)
├── gemm_conv1x1/
│   ├── aie2_gemm_conv1x1.py         # IRON gen for 1×1 conv via GEMM
│   ├── gemm_conv1x1_pythoc.py       # Standalone single-shape PythoC GEMM example
│   └── spot_check.py                # Single-shape correctness check (debug)
├── kernels/
│   ├── build_kernels.py             # Compiles PythoC .cc kernels to .o
│   ├── rep_elan_bf16.cc             # The C kernel sources (3x3, gemm, k-blocked, residual)
│   ├── rep_elan_bf16_pythoc.py      # PythoC wrapper for kernel compilation
│   └── build/                       # .o outputs (gitignored)
├── graphs/
│   └── export_mdv6_graphs.py        # Dump model topology to .onnx / .html / .json
├── prototypes/                      # Per-layer standalone PythoC examples — NOT used by the
│   ├── README.md                    # production path; reference for porting new layers
│   ├── lit.local.cfg                # .py suffix override (each prototype is a lit test)
│   └── <9 dirs>/<layer>_pythoc.py   # aconv, batchnorm_silu, bottleneck, conv, elan,
│                                    # elementwise, repconv, repncsp, repncsp_elan, sppelan
└── *.md                             # PHASE_*_PLAN, PHASE_E_BOTTLENECK_MODEL, CONVERSION_PATTERN,
                                      # PHASE_H_DEVICE_STITCHING_PLAN, IMPL_ARCH_STATUS (this)
```

## Testing gaps

The full-model test (`test_full_model_mc.py`) provides end-to-end PASS/FAIL
coverage across every layer. **Standalone bytewise correctness tests
exist only for a subset** of the merged ELFs — the others rely on the
full-model test to catch regressions, which means a per-ELF bug can be
masked by downstream layer interactions before the detection head.

What's tested standalone (in `conv/test_*.py`):

| ELF | Test |
|---|---|
| `merged_gemm_t164_ic96_oc48_p1_pair_x1` (re6_rn1) | `test_pair_rn1.py` |
| `ocb_re6_rn3_x1` | `test_ocb_re6_rn3.py` |
| `ocb_re8_rn3_x1` | `test_ocb_re8_rn3.py` |

What's NOT tested standalone (relies on full-model coverage):

| ELF | Why no test |
|---|---|
| `ocb_re4_c3_x1`, `ocb_re6_c3_x1`, `ocb_re8_c3_x1` | Phase F — never built standalone tests |
| `ocb_aconv3_x1`, `_aconv7_x1`, `_aconv16_x1`, `_aconv19_x1` | Phase G — same |
| `merged_re4_rn3_p4_x1` | Pre-OCB single-clone, never tested standalone |
| `merged_aconv5_p4_x1` | Same |
| `merged_ftconv0_x8`, `merged_ftconv1_p2_x4`, `merged_elan_c3_p4_x4` | Phase A.1 fanouts — old standalone tests were deleted because they validated against the legacy xclbin baseline which is gone |
| `merged_gemm_t256_ic64_oc32_p1_pair_x1` (re4_rn1), `merged_gemm_t104_ic128_oc64_p1_pair_x1` (re8_rn1) | Phase C step A — only re6_rn1 has a test |
| GEMM K-blocked ELFs (`merged_gemm_*_kbN_*`) | Never had a standalone test |

Cheapest gap-fillers (in priority order):

1. Generalize `test_ocb_re6_rn3.py` to parameterize on layer name + shape;
   instantiate for every OCB layer (one test file → 8 test invocations).
2. Generalize `test_pair_rn1.py` similarly for the 3 GEMM pair ELFs.
3. New `test_merged_fanout.py` validating x4/x8 fanout ELFs against
   reference computed in numpy/torch (not xclbin baseline).
4. K-blocked GEMM correctness: build a single-shape standalone test.

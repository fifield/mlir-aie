# MDV6 AIE2P Performance Optimization Plan

## Current State

Full MDV6-mit-yolov9-c forward pass on Strix Halo NPU.

### Post-Phase-A + sub-task B1 (2026-04-18)

Warm-frame profile (`test_full_model_mc.py --profile 7`):

```
wall             1869 ms  (0.54 fps — video-stream target throughput)
  npu_run        1103 ms  (59%)   — NPU active in DefaultNPURuntime.run
  launch_gap      393 ms  (21%)   — per-call pyxrt plumbing (867 µs × 453 launches)
  pre_post        293 ms  (16%)   — model setup + Detection (CPU) pre/post
  cpu_layers       52 ms  ( 3%)   — RepConv 33 + Detection 11 + AvgPool 7 + Upsample 1
  numpy            34 ms  ( 2%)   — np.concatenate (host weight assembly)
  iron_alloc        0 ms  ( 0%)   — eliminated by buffer pool (0pf-A)
  pack/fuse         0 ms  ( 0%)   — essentially free after warmup
launches         453 (down from 607 pre-optimization)
```

Correctness: `max_class_diff=0.2264, max_vector_diff=0.0312, PASS`.

Multi-frame verified: 5 consecutive `main()` calls all PASS with identical
class diff — video-stream inference safe (see bead `mlir-aie-woi`).

### Cumulative progress (0pf beads)

| stage | wall | npu_run | iron_alloc | launch_gap | launches | fps | bead |
|---|---:|---:|---:|---:|---:|---:|---|
| before 0pf | 2248 | 1275 | 183 | 381 | 567 | 0.44 | — |
| after 0pf-A (buffer pool) | 2081 | 1269 | **0** | 392 | 567 | 0.48 | mlir-aie-0pf |
| after 0pf-B1 (tile-count ppc) | **1869** | **1103** | 0 | 393 | **453** | **0.54** | mlir-aie-0pf |
| Δ cumulative | **-379 ms** | **-172** | **-183** | +12 | **-114** | **+23%** | |

### 32-core multicore (2026-03-28, earlier)
- **6.0s** total, 32 cores, scalar kernels — **24× over single-core baseline**
- Correctness bug (zeros in output) **fixed** 2026-04-17 — root causes were:
  1. `mlir-aie-9dl` — ppc>1 ObjectFifo split in `aie2_gemm_conv1x1.py`
  2. `mlir-aie-woi` — multi-frame id-recycle collisions in `fuse_bn` cache
  3. L1-budget overflow in vectorized 3×3 at large tile dims
- After those fixes: 3.2s single-pass, then 1.87s after host-pipeline work

### Single-core baseline (2026-03-17, historical)
- **145.5s** total
- **0.25 GFLOP/s** effective (0.005% of per-tile peak)
- 34 GFLOP total model compute

### 32-core per-layer timing (pre-optimization, 2026-03-28)
| Layer | SC time | MC time | Speedup |
|-------|---------|---------|---------|
| conv0 | 3.3s | 0.3s | 11× |
| conv1 | 6.8s | 0.2s | 34× |
| elan2 | 15.1s | 0.5s | 30× |
| aconv3 | 5.8s | 0.2s | 29× |
| rep_elan4 | 24.0s | 0.8s | 30× |
| aconv5 | 5.3s | 0.2s | 27× |
| rep_elan6 | 15.0s | 0.6s | 25× |
| aconv7 | 2.8s | 0.2s | 14× |
| rep_elan8 | 7.9s | 0.5s | 16× |
| spp9 | 2.1s | 0.1s | 21× |
| rep_elan12 | 16.6s | 0.5s | 33× |
| rep_elan15 | 26.5s | 0.8s | 33× |
| head P4 | 9.5s | 0.6s | 16× |
| head P5 | 4.9s | 0.4s | 12× |
| **Total** | **145.5s** | **6.0s** | **24×** |

### Operator-level Profiling

All operators are **compute-bound** (arithmetic intensity 20-340).

| Operator | ms/tile | GFLOP/s | AI | Notes |
|----------|---------|---------|-----|-------|
| Conv1x1 (any size) | 5-14 | 0.10-0.12 | 20-120 | mmul-friendly |
| Conv3x3 (any size) | 2-17 | 0.25-0.30 | 140-340 | highest AI |
| Conv3x3 stride=2 | 1-8 | 0.21-0.29 | 190-270 | AConv inner conv |

All are **500-1600× below per-tile peak** (160 GFLOP/s bf16 mmul @ 1.25 GHz;
see AIE2P ArchSpec Table 2-1: 0.16 TOPS = 64 bf16 multipliers × 2 ops/MAC × 1.25 GHz).

---

## Hardware Model

### Strix Halo AIE2P (6 rows × 8 columns)

Device model: `npu2` (xrt-smi reports topology `6x8`)

```
        Col 0     Col 1     Col 2     Col 3     Col 4     Col 5     Col 6     Col 7
       ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
Row 5  │Compute │Compute │Compute │Compute │Compute │Compute │Compute │Compute │
Row 4  │Compute │Compute │Compute │Compute │Compute │Compute │Compute │Compute │  4 compute
Row 3  │Compute │Compute │Compute │Compute │Compute │Compute │Compute │Compute │  tiles per
Row 2  │Compute │Compute │Compute │Compute │Compute │Compute │Compute │Compute │  column
       ├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
Row 1  │MemTile │MemTile │MemTile │MemTile │MemTile │MemTile │MemTile │MemTile │  512KB each
       ├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
Row 0  │  Shim  │  Shim  │  Shim  │  Shim  │  Shim  │  Shim  │  Shim  │  Shim  │  Host DMA
       └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

**Per tile:**
- Compute: 64KB data memory, 16KB program memory, 2 DMA (1 S2MM + 1 MM2S)
- bf16 mmul<4,8,8>: 160 GFLOP/s peak (0.16 TOPS @ 1.25 GHz, 64 multipliers)

**Per column:**
- Shim: 4 DMA channels (2 S2MM + 2 MM2S)
- Memtile: 512KB, 12 DMA channels (6 S2MM + 6 MM2S)
- Compute: 4 tiles × 64KB, 2 DMA each (BD-chainable for multiplexing)

**Array totals:**
- 32 usable compute tiles (8 cols × 4 tiles)
- ~5,120 GFLOP/s peak (bf16 mmul, 32 tiles × 160 GFLOP/s)
- 8 × 512KB = 4MB L2 (memtile)

### Roofline

| | GFLOP/s | % of peak | Time for 34 GFLOP |
|---|---------|-----------|-------------------|
| 1 tile, scalar | 0.25 | 0.16% | 145s (historical) |
| 1 tile, vectorized (mmul+BN) | 4-16 | 2.5-10% | 2-9s |
| 1 tile, vectorized (mmul+BN+SiLU) | 2-4 | 1.3-2.5% | 9-17s |
| 1 tile, peak | 160 | 100% | 0.2s |
| 32 tiles, vectorized | 64-512 | 1.3-10% | 0.07-0.5s |
| Array peak | 5,120 | 100% | 6.6ms |

Current wall is dominated by `npu_run` (1103 ms) + `launch_gap` (393 ms).
At ~30 GFLOP useful compute / 1103 ms ≈ **27 GFLOP/s sustained on the array**
(vs 5120 peak) — the NPU is still ~190× under array peak, so Phase A
kernel gains remain plausible.

---

## Host Pipeline Characterization (bead `mlir-aie-0pf`)

### Sub-task A: XRT buffer pool — ✅ DONE (commit 73d8925d4)

Previously the full model allocated 1491 `iron.tensor`/`iron.zeros` per warm
frame (924 inputs + 567 weights + 567 outputs) at 130-150 µs each — **~183 ms
of pyxrt buffer construction per frame, eliminated to 0 ms**.

Key gotchas in `run_tiled_mc.py`:
- **Role-separated pools** (`_INPUT_POOL`, `_WEIGHT_POOL`, `_OUTPUT_POOL`):
  A 1×1 conv with `ic == oc` (e.g., elan_c1) would otherwise alias input
  and output to the same XRT buffer → kernel hangs.
- **Direct writes to `.data` + explicit `_sync_to_device()`**: going through
  `buf.numpy()` would unconditionally sync_from_device first (clobbering the
  pending write with stale device data) and never sync back — silent
  corruption that looks like a bug in the kernel.

### Sub-task B1: ppc tile-count rule — ✅ DONE (commit cc511011d)

The only ppc bumps that help are ones where `calls_per_ocb` actually drops:

```
Rule: bump ppc on a layer only when
    n_tiles > N_CORES × prior_ppc

Otherwise calls_per_ocb = ceil(n_tiles / (N_CORES × ppc)) is unchanged
and the core just runs `ppc` padded iterations per call — pure regression.
```

Applied bumps (validated by deterministic launch-count delta under `--profile 7`):

| layer | n_tiles | ppc | calls@ppc=1 → after |
|---|---:|---:|---:|
| mc_ftconv1 | 196 | **2** | 7 → 4 per ocb |
| mc_elan_c3 | 400 | **4** | 13 → 4 |
| mc_aconv5 | 100 | **4** | 4 → 1 |
| mc_re4_c3 | 49 | **2** | 2 → 1 |
| mc_re4_rn3 | 100 | **4** | 4 → 1 |

Layers NOT bumped (n_tiles ≤ 32, would be regression):
mc_aconv7/16/19, mc_re6_c3/rn3, mc_re8_c3/rn3, mc_re4_c4, mc_re6_c1/c4/rn1/rnm,
mc_re8_c1/c4/rn1/rnm, mc_spp_c1, mc_re12_c1, mc_re15_*.

**Implication for other N_CORES**: the 32 in the rule is `N_CORES`. At
fewer cores (e.g., single-column 4-tile runs), the threshold drops
proportionally and more layers become bumpable — see the inline table in
`run_tiled_mc.py::_MC_PPC` for the full per-layer analysis.

### Sub-tasks NOT pursued (bead revisions 2026-04-18)

- **mlir-aie-d6f (cache packed weights)** — downgraded **P1 → P3**. Original
  "~500 ms savings" estimate was based on cold-frame measurements. Actual
  warm-frame `pack` bucket is ~0 ms; prewarm + WeakKeyDict cache already
  addresses it.
- **mlir-aie-cup (RepConv to NPU)** — downgraded **P1 → P3**. Original
  "~600 ms" estimate was wrong by an order of magnitude. Measured
  `cpu.RepConv = 33 ms` in the warm baseline. Not worth the kernel
  complexity at current spatial resolutions.
- **0pf-B sub-task B (launch batching, host-side plumbing tricks)** —
  the `launch_gap = 393 ms` bucket is dominated by per-call
  `_sync_to_device` + `_sync_from_device` (scales linearly with buffer
  size × launch count). Total bytes moved is invariant; only the fixed
  ~200 µs pyxrt overhead per launch is reducible. With B1's launch-count
  reduction, the remaining fixed-overhead surface is small. Further
  savings require **on-chip layer chaining** (Phase C) to eliminate the
  round-trip entirely.

### Profile harness

`profile_harness.py` + `--profile N` on `test_full_model_mc.py` run N
forward passes (first warm-up, rest measured). Monkey-patches
`DefaultNPURuntime.load/.run`, `iron.tensor/zeros`, repacking helpers,
fuse_bn variants, `np.concatenate`, and CPU-layer classes. Emits an
8-bucket breakdown with sub-bucket detail. `--save-baseline` writes a
JSON reference; `--baseline baseline.json` fails on >10% category
regression.

Any host-pipeline change should run `--profile` and either update
`profile_baseline.json` or prove no regression. Without this we made
order-of-magnitude wrong estimates (see bead history for d6f/cup
downgrades and 0pf-B1 wall-time bisect that was initially noise-blinded).

---

## DMA Channel Budget (verified)

### Per column, 4 compute tiles, spatial parallel with weight broadcast

```
Shim (4 channels):
  MM2S #0: Weights → memtile                    (used)
  MM2S #1: Input patches → memtile              (used)
  S2MM #0: Output tiles ← memtile               (used)
  S2MM #1: (spare)
  = 3/4 ✓

Memtile (12 channels):
  MM2S #0: Weight broadcast → 4 compute tiles   (used, 1 DMA, multicast)
  MM2S #1-4: Input patch → compute tile 0-3     (used, 4 DMAs, unicast)
  S2MM #0-3: Output tile ← compute tile 0-3     (used, 4 DMAs)
  S2MM #4-5: (spare)
  MM2S #5: (spare)
  = 9/12 ✓ (3 spare channels available for L2 staging or double-buffering)

Compute tile (2 channels each):
  S2MM #0: Weight + Input via BD chaining        (used, time-multiplexed)
  MM2S #0: Output tile                           (used)
  = 2/2 ✓
```

**Weight broadcast** saves 3 memtile DMA channels vs per-tile unicast.
All spatial tiles within the same oc_block share identical weights.

**Spare channels (3 per column)** can be used for:
- Double-buffering input/output for latency hiding
- L2 staging for operator chaining (Level 4)
- Skip connection writes to external

---

## L1 Memory Verification

All operators fit in 64KB with input patch + weights + output tile + 4KB stack.

| Operator | Patch | Weights | Output | Total | Fits? |
|----------|-------|---------|--------|-------|-------|
| re4 conv1x1 128→128 (t=10) | 25.0KB | 16.2KB | 12.5KB | 57.8KB | ✓ |
| re4 conv3x3 64→64 (t=12) | 24.5KB | 18.1KB | 4.5KB | 51.1KB | ✓ |
| re4 conv1x1 256→128 (t=8) | 32.0KB | 16.1KB | 4.0KB | 56.1KB | ✓ |
| re4rn conv3x3 32→32 (t=16) | 20.2KB | 18.1KB | 16.0KB | 58.4KB | ✓ (worst) |
| elan conv1x1 64→64 (t=8) | 8.0KB | 8.2KB | 8.0KB | 28.2KB | ✓ |
| re6 conv3x3 96→96 (t=8) | 18.8KB | 27.1KB | 2.0KB | 51.8KB | ✓ |
| re8 conv3x3 128→128 (t=4) | 9.0KB | 36.1KB | 0.5KB | 49.6KB | ✓ |

---

## Dataflow: Data Residency

### Inter-layer activation sizes

| From → To | Size | Residency |
|-----------|------|-----------|
| conv0 → conv1 | 6.4MB | External |
| conv1 → elan2 | 3.2MB | External |
| elan2 → aconv3 | 3.2MB | External |
| aconv3 → re4 | 1.6MB | External |
| re4 → aconv5 + **B3 skip** | 1.6MB | External (skip persists) |
| aconv5 → re6 | 614KB | L2 possible (tight) |
| re6 → aconv7 + **B4 skip** | 614KB | External (skip persists) |
| aconv7 → re8 | **200KB** | **L2 ✓** |
| re8 → spp9 | **200KB** | **L2 ✓** |
| spp9 → up + **N3 skip** | **200KB** | **L2 ✓** / External for skip |
| re12 → up + **N4 skip** | 614KB | External (skip persists) |
| re15 → head | 1.6MB | External |
| re21 → detect | **200KB** | **L2 ✓** |

### L2 dataflow chains (no external round-trip)

**20×20 chain:** aconv7 (200KB) → re8 (200KB) → spp9 (200KB) → re21 (200KB)
All fit in 512KB memtile. Data stays on-chip for 4 consecutive layers.

### Within RepNCSPELAN sub-operators

Each RepNCSPELAN is 6 sequential sub-ops. For 20×20 layers, all intermediate
activations (≤200KB) can stage through L2 memtile:

```
Conv1x1 ──L2──→ RepNCSP ──L2──→ Conv3x3 ──L2──→ RepNCSP ──L2──→ Conv3x3 ──L2──→ Conv1x1
```

For 80×80 layers (1.6MB intermediates), must go external between sub-ops.

### Skip connections (always external)

| Skip | Size | Source → Dest | Layers skipped |
|------|------|---------------|----------------|
| B3 | 1.6MB | re4 → re15 | 5 |
| B4 | 614KB | re6 → re12 | 3 |
| N3 | 200KB | spp9 → re21 | 5 |
| N4 | 614KB | re12 → re18 | 2 |

Strategy: Write skip data to external via DMA during next layer's compute (overlap).

---

## Optimization Tiers

### Tier 1: Vectorize + reduce invocations — Phase A/B, partially shipped

Single-core kernel work + host-pipeline invocation reduction.

| Action | Speedup | Status |
|--------|---------|--------|
| mmul<4,8,8> for matmul core | 10-15× on matmul | ✅ (conv1x1 + conv3x3) |
| Vector BN (aie::mul + aie::add) | 2× on BN portion | ✅ |
| Scalar SiLU (blocked: no bf16→float vec) | 1× (unchanged) | bead `mlir-aie-4zz` |
| Larger tiles (12-16 vs 8) | 2-4× fewer invocations | ✅ partial (per-layer L1 cap) |
| ppc batching (tile-count rule) | 1-4× fewer calls per layer | ✅ (bead `mlir-aie-0pf` B1) |
| Pre-transposed weights (contiguous loads) | 2-3× on weight load | ✅ (conv1x1) |
| **Combined (today)** | **host+kernel wins: 77× from baseline** | 145.5s → 1.87s |

### Tier 2: Multi-core spatial parallel — ✅ DONE

32 tiles (8 cols × 4 tiles) process same layer in parallel.
Current measured: **1.87s warm-frame** (0.54 fps).

### Tier 3: On-chip pipeline — Phase C, not started

Layers mapped to different columns, data flows tile-to-tile.

```
Col 0-1 (8 tiles):  conv0 + conv1 + elan2 (large spatial, few channels)
Col 2-3 (8 tiles):  aconv3 + re4 + aconv5 (mid backbone)
Col 4-5 (8 tiles):  re6 + aconv7 + re8 + spp9 (deep backbone)
Col 6-7 (8 tiles):  neck + head (re12, re15, re18, re21, detect)
```

- Pipeline latency: ~50ms (one frame)
- Pipeline throughput: ~12ms/frame (~80 FPS) for streaming
- 8 columns allows more balanced pipeline stages

Beads `mlir-aie-2vb` (epic), `mlir-aie-9oz` (Flavor 1: L2 memtile chains),
`mlir-aie-k82` (Flavor 2: cross-column spatial pipeline).

This is the only remaining lever to meaningfully reduce the 393 ms
`launch_gap` bucket: current host-pipeline work has amortised the fixed
per-launch overhead as far as measurement resolution allows; the ~400 µs
DMA sync per call is a hard floor unless the round-trip is eliminated.

---

## Test Plan

Bottom-up validation, each level proves a capability before scaling.

### Level 0: Single tile, scalar ✅ DONE
- All 10 layer types pass on NPU at 8×8 test dims
- Full model end-to-end pass at model dims (145.5s historical)

### Level 1: Single tile, vectorized ✅ DONE
- conv1x1 vec: mmul<4,8,8> + pre-transposed weights + vector BN
- conv3x3 vec: mmul<4,8,8> per kernel position + vector BN
- 3.2× speedup (limited by scalar SiLU)
- 15.4× speedup without SiLU

### Level 2: 2-tile weight broadcast ✅ DONE (`mlir-aie-kot`)
- 2 Workers on same column, per-core ObjectFifos + TensorAccessPattern
- 1 weight ObjectFifo → 2 consumers (broadcast)
- Conv1x1 16→16, tile 8×8: max_diff=0.125, 1.8ms

### Level 3: 4-tile full column ✅ DONE (`mlir-aie-03i`)
- 4 Workers, full column utilization (rows 2-5)
- Per-column: bulk input FIFO → split at memtile → 4 cores, join output
- Per-column weight FIFO → broadcast to 4 consumers

### Level 4: Operator chain L1→L2→L1 ✅ DONE (`mlir-aie-0m4`)
- 2 conv1x1 Workers chained: Worker1 output → Worker2 input (same column)
- Intermediate ObjectFifo stays on-chip (no external round-trip)

### Level 5: 8-column 32-tile spatial ✅ DONE (`mlir-aie-646`)
- Full array (`npu2`), all 8 columns active, 32 cores total
- Hierarchical: per-column split/join at memtile + per-column weight broadcast
- `aie2_multicore_broadcast.py` scales 1-32 cores automatically

### Level 6: Full model 32-core ✅ DONE (timing AND correctness)
- All 14 layers complete; correctness passes after `mlir-aie-9dl` and
  `mlir-aie-woi` fixes (2026-04-17/18)
- Currently 1.87s warm-frame wall, 0.54 fps

### Level 5b: Skip connections + DMA overlap (`mlir-aie-xdg`)
- B3/B4/N3/N4 write to external during compute
- Not yet started (lower priority than Phase A kernel work)

---

## Beads Tracking

### Epic & top-level
| Bead | Title | Status |
|------|-------|--------|
| `mlir-aie-mi7` | Perf optimization epic | Open |
| `mlir-aie-326` | Phase A: Vectorized kernels | In progress |
| `mlir-aie-1wy` | Phase B: Multi-core spatial | Done (timing+correctness) |
| `mlir-aie-2vb` | Phase C: On-chip pipelining (epic) | P3, not started |
| `mlir-aie-9oz` | Phase C Flavor 1: L2 memtile chains | P3, not started |
| `mlir-aie-k82` | Phase C Flavor 2: cross-column pipeline | P3, not started |
| `mlir-aie-9xq` | Phase D: RepConv/AvgPool/Upsample to NPU | P2, not started |

### Test-plan levels
| Bead | Title | Status |
|------|-------|--------|
| `mlir-aie-kot` | Level 2: 2-tile broadcast | ✅ Done |
| `mlir-aie-03i` | Level 3: 4-tile column | ✅ Done |
| `mlir-aie-0m4` | Level 4: L1→L2→L1 chain | ✅ Done |
| `mlir-aie-646` | Level 5: 32-tile spatial | ✅ Done |
| `mlir-aie-zuq` | Level 6: Full model 32-core | ✅ Done (timing + correctness) |
| `mlir-aie-xdg` | Level 5b: Skip connections + DMA overlap | P2, ready |

### Correctness + stability (closed)
| Bead | Title | Status |
|------|-------|--------|
| `mlir-aie-9dl` | Fix ppc>1 ObjectFifo split in aie2_gemm_conv1x1.py | ✅ Closed 2026-04-17 |
| `mlir-aie-woi` | Multi-frame inference fails: stale cached weights from re-used Python id | ✅ Closed 2026-04-18 |

### Host pipeline (0pf family)
| Bead | Title | Status |
|------|-------|--------|
| `mlir-aie-0pf` | Batch Bottleneck calls + reduce per-call Python dispatch | P2, sub-task A+B1 done |
| `mlir-aie-d6f` | Cache packed GEMM/conv weights across NPU calls | P3, already mostly addressed |
| `mlir-aie-cup` | Move RepConv fused dual-path to NPU | P3, ROI too small at current spatial dims |

### Known bugs / open items
| Bead | Title | Status |
|------|-------|--------|
| `mlir-aie-mi7.1` | MC correctness bug (zeros) | **Resolved** by 9dl + woi + L1 fixes |
| `mlir-aie-mi7.1.1` | Minimal reproducer: DMA buffer overflow crashes NPU | P1, open |
| `mlir-aie-mi7.2` | aiecc.py: ObjectFifo naming affects compiled ELF correctness | P1, open |
| `mlir-aie-4zz` | Vectorize bf16 SiLU | P2, open (the last kernel op not vectorized) |

---

## Open Questions / Next-Step Options

Ordered by expected wall-time impact at current 1.87s wall:

1. **Phase A completion** (`mlir-aie-4zz`) — vectorise bf16 SiLU. Blocked on
   a bf16→float vector conversion in the kernel toolchain. Potentially
   significant if SiLU is a measurable chunk of `npu_run`. Needs per-layer
   NPU-time profiling to size the prize (profile harness does not break
   down NPU time internally).

2. **Phase C Flavor 1** (`mlir-aie-9oz`) — L2 memtile dataflow chains for
   20×20 sequences (aconv7→re8→spp9→re21). Removes external round-trips,
   directly shrinks `launch_gap`. Medium complexity; clear spec in docs.

3. **Phase D** (`mlir-aie-9xq`) — Detection on NPU would save ~11 ms; other
   CPU layers (RepConv 33, AvgPool 7, Upsample 1) total <60 ms. Low ROI
   relative to Phase C.

4. **Regression guard** — extend `test_full_model_mc.py --baseline` to CI.
   Without an automated regression check, subtle host-pipeline edits can
   silently undo the 0pf wins (see B1 sweep analysis notes in the
   `_MC_PPC` comment).

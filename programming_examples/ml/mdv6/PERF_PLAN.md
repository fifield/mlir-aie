# MDV6 AIE2P Performance Optimization Plan

## Current State

Full MDV6-mit-yolov9-c forward pass on Strix Halo NPU.

### 32-core multicore (2026-03-28)
- **6.0s** total, 32 cores, scalar kernels — **24× over single-core baseline**
- Correctness: NaN in output (bug in MC weight/patch packing, see `mlir-aie-mi7.1`)
- 28 unique MC xclbins, hierarchical split/join + per-column weight broadcast
- `test_full_model_mc.py` runs all 14 layers end-to-end

### Single-core baseline (2026-03-17)
- **145.5s** total (single-core scalar kernels, host-orchestrated tiling)
- **0.25 GFLOP/s** effective (0.03% of peak)
- 34 GFLOP total model compute
- ~10,000 host→NPU round-trips (1 per spatial tile)

### 32-core per-layer timing
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

### Time Breakdown

| Layer | Spatial | Time | % | FLOPs | Tiles |
|-------|---------|------|---|-------|-------|
| re15 (RepNCSPELAN) | 80×80 | 26.5s | 18% | 6.0G | ~1000 |
| re4 (RepNCSPELAN) | 80×80 | 24.0s | 16% | 5.2G | ~700 |
| re12 (RepNCSPELAN) | 40×40 | 16.6s | 11% | 4.5G | ~1000 |
| elan2 (ELAN) | 160×160 | 15.1s | 10% | 1.1G | ~1000 |
| re6 (RepNCSPELAN) | 40×40 | 15.0s | 10% | 3.8G | ~1000 |
| head_p4 | 40×40 | 9.5s | 7% | 4.8G | ~1000 |
| re8 (RepNCSPELAN) | 20×20 | 7.9s | 5% | 2.2G | ~600 |
| conv1 | 320×320 | 6.8s | 5% | 0.9G | 784 |
| aconv3 | 80×80 | 5.8s | 4% | 0.9G | 800 |
| head_p5 | 20×20 | 4.9s | 3% | 2.8G | ~600 |
| conv0 | 640×640 | 3.3s | 2% | 0.2G | 196 |
| aconv5 | 40×40 | 5.3s | 4% | 0.7G | 2400 |
| aconv7 | 20×20 | 2.8s | 2% | 0.4G | 1600 |
| spp9 | 20×20 | 2.1s | 1% | 0.4G | ~100 |
| detect | multi | <0.1s | 0% | 0.05G | CPU |

### Operator-level Profiling

All operators are **compute-bound** (arithmetic intensity 20-340).

| Operator | ms/tile | GFLOP/s | AI | Notes |
|----------|---------|---------|-----|-------|
| Conv1x1 (any size) | 5-14 | 0.10-0.12 | 20-120 | mmul-friendly |
| Conv3x3 (any size) | 2-17 | 0.25-0.30 | 140-340 | highest AI |
| Conv3x3 stride=2 | 1-8 | 0.21-0.29 | 190-270 | AConv inner conv |

All are **100-1000× below per-tile peak** (~16 GFLOP/s bf16 mmul).

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
- bf16 mmul<4,8,8>: ~16 GFLOP/s peak

**Per column:**
- Shim: 4 DMA channels (2 S2MM + 2 MM2S)
- Memtile: 512KB, 12 DMA channels (6 S2MM + 6 MM2S)
- Compute: 4 tiles × 64KB, 2 DMA each (BD-chainable for multiplexing)

**Array totals:**
- 32 usable compute tiles (8 cols × 4 tiles)
- ~512 GFLOP/s peak (bf16 mmul, 32 tiles × 16 GFLOP/s)
- 8 × 512KB = 4MB L2 (memtile)

### Roofline

| | GFLOP/s | Time for 34 GFLOP |
|---|---------|-------------------|
| 1 tile, scalar | 0.25 | 145s (current) |
| 1 tile, vectorized (mmul+BN) | 4-16 | 2-9s |
| 1 tile, vectorized (mmul+BN+SiLU) | 2-4 | 9-17s |
| 32 tiles, vectorized | 64-512 | 0.07-0.5s |
| Array peak | 512 | 0.066s = 66ms |

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

### Tier 1: Vectorize + reduce invocations → ~15s (10×)

Single core, but faster kernels and fewer host round-trips.

| Action | Speedup | Status |
|--------|---------|--------|
| mmul<4,8,8> for matmul core | 10-15× on matmul | Done (conv1x1) |
| Vector BN (aie::mul + aie::add) | 2× on BN portion | Done |
| Scalar SiLU (blocked: no bf16→float vec) | 1× (unchanged) | Blocked |
| Larger tiles (12-16 vs 8) | 2-4× fewer invocations | Partially done |
| n_patches batching | 2-4× fewer invocations | Not started |
| Pre-transposed weights (contiguous loads) | 2-3× on weight load | Done (conv1x1) |
| **Combined** | **~10×** | |

### Tier 2: Multi-core spatial parallel → ~0.5-2s (100×)

32 tiles (8 cols × 4 tiles) process same layer in parallel.

| Layer group | Current | 32-core est. | Speedup |
|------------|---------|-------------|---------|
| 80×80 layers | 50.5s | ~0.3s | 170× |
| 160×160 layers | 15.1s | ~0.12s | 126× |
| 40×40 layers | 41.1s | ~0.4s | 103× |
| 20×20 layers | 12.1s | ~0.14s | 86× |
| Stride-2 convs | 17.1s | ~0.2s | 86× |
| **Total** | **145.5s** | **~1.5s** | **97×** |

With vectorized kernels on top: **~0.5s** (290×).

### Tier 3: On-chip pipeline → <100ms (1000×+)

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

---

## Test Plan

Bottom-up validation, each level proves a capability before scaling.

### Level 0: Single tile, scalar ✅ DONE
- All 10 layer types pass on NPU at 8×8 test dims
- Full model end-to-end pass at model dims (145.5s)

### Level 1: Single tile, vectorized ✅ DONE
- conv1x1 vec: mmul<4,8,8> + pre-transposed weights + vector BN
- conv3x3 vec: mmul<4,8,8> per kernel position + vector BN
- 3.2× speedup (limited by scalar SiLU)
- 15.4× speedup without SiLU

### Level 2: 2-tile weight broadcast ✅ DONE (`mlir-aie-kot`)
- 2 Workers on same column, per-core ObjectFifos + TensorAccessPattern
- 1 weight ObjectFifo → 2 consumers (broadcast)
- Conv1x1 16→16, tile 8×8: max_diff=0.125, 1.8ms
- **Architecture**: per-core FIFOs with TAP (not split/join) for multi-column support

### Level 3: 4-tile full column ✅ DONE (`mlir-aie-03i`)
- 4 Workers, full column utilization (rows 2-5)
- Per-column: bulk input FIFO → split at memtile → 4 cores, join output
- Per-column weight FIFO → broadcast to 4 consumers
- Conv1x1 16→16, tile 8×8: max_diff=0.125, 1.6ms
- Shim DMA: 3/4 channels (1 weight + 1 input + 1 output)

### Level 4: Operator chain L1→L2→L1 ✅ DONE (`mlir-aie-0m4`)
- 2 conv1x1 Workers chained: Worker1 output → Worker2 input (same column)
- Intermediate ObjectFifo stays on-chip (no external round-trip)
- Conv1x1 16→16 × 2, tile 8×8: max_diff=0.5 (cumulative error), 12.1ms

### Level 5: 8-column 32-tile spatial ✅ DONE (`mlir-aie-646`)
- Full array (`npu2`), all 8 columns active, 32 cores total
- Hierarchical: per-column split/join at memtile + per-column weight broadcast
- Conv1x1 16→16, tile 8×8: max_diff=0.125, 4.1ms (32 patches)
- Conv1x1 16→16, tile 10×10, 80×80 layer: 7.5ms (64 tiles in 2 invocations)
- **Key pattern**: `aie2_multicore_broadcast.py` scales 1-32 cores automatically

### Level 6: Full model 32-core ✅ TIMING DONE, CORRECTNESS WIP (`mlir-aie-zuq`)
- All 14 layers complete in **6.0s** (24× over 145.5s baseline)
- 28 unique MC xclbins (deduplicated from 34 to fit 32-slot XRT cache)
- Architecture: per-column split/join + per-column weight broadcast + TensorAccessPattern
- RepNCSP sub-layers also use MC (RepConv still on CPU)
- Bug: MC path returns zeros for some layers → NaN in detection output
  - SC path produces correct non-zero output for same inputs
  - Root cause: likely weight packing or patch extraction in `_run_tiled_mc_inner`
- Fixed: mc_re4_rn3 buffer mismatch (tile=12→16, oc=16→32) that was crashing NPU
- Fixed: mc_re6_rnm buffer mismatch (ic=192→96)

### Level 5b: Skip connections + DMA overlap (`mlir-aie-xdg`)
- B3/B4/N3/N4 write to external during compute
- Concurrent DMA + compute validation
- Not yet started (lower priority than L6 correctness)

---

## Beads Tracking

| Bead | Title | Status |
|------|-------|--------|
| mlir-aie-mi7 | Perf optimization epic | Open |
| mlir-aie-326 | Phase A: Vectorized kernels | In progress |
| mlir-aie-1wy | Phase B: Multi-core spatial | In progress |
| mlir-aie-9xq | Phase D: RepConv/AvgPool/Upsample to NPU | Open |
| mlir-aie-kot | Level 2: 2-tile broadcast | ✅ Done |
| mlir-aie-03i | Level 3: 4-tile column | ✅ Done |
| mlir-aie-0m4 | Level 4: L1→L2→L1 chain | ✅ Done |
| mlir-aie-646 | Level 5: 32-tile spatial | ✅ Done |
| mlir-aie-zuq | Level 6: Full model 32-core | ⚠️ 6.0s timing done, correctness WIP |
| mlir-aie-mi7.1 | MC correctness bug (zeros) | Open P1 |
| mlir-aie-xdg | Level 5b: Skip connections | Ready |

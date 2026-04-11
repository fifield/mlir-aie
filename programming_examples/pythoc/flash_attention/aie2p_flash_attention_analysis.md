# AIE2P Flash Attention — Architecture & Implementation Analysis

> **Source**: `mlir-air/programming_examples/flash_attention/dataflow_based/attn.py`
> **Generated AIR MLIR**: `build_peano/air.mlir`
> **Generated AIE MLIR**: `build_peano/air_project/aie.air.mlir`
> **Target**: `aie.device(npu2)` (AIE2P / Strix Point)

---

## 1. Algorithm Overview

The implementation is the **online (streaming) flash attention** algorithm operating on
bf16 data. It computes `softmax(Q·K^T) · V` without materializing the full attention
matrix, processing K/V in chunks of size `lkp` and maintaining running softmax
statistics (`up` = running max, `sp` = running sum of exponentials).

### Reference Python (from `attn.py` lines 780-801)

```python
Gp = zeros(lq, dv)          # running weighted-value accumulator
up = full(lq, 1, -inf)      # running row-max
sp = zeros(lq, 1)           # running row-sum-of-exp

for j in range(lk // lkp):
    G  = Q @ K_chunk + mask  # [lq, lkp]  — score tile
    u  = rowmax(G)           # [lq, 1]
    u  = max(u, up)          # new running max
    G  = exp(G - u)          # softmax numerator
    r  = exp(up - u)         # rescale factor for old accumulator
    Gp = Gp * r              # rescale old accumulator
    Gp = G @ V_chunk + Gp    # accumulate new weighted values
    s  = rowsum(G)           # [lq, 1]
    s  = sp * r + s          # update running sum
    sp, up = s, u

output = Gp / sp             # final normalization
```

---

## 2. Default Problem Dimensions

| Parameter | Makefile default | Python func default | Meaning |
|-----------|-----------------|---------------------|---------|
| `lk`      | 12288           | 3072                | Total K/V sequence length |
| `lkp`     | 96              | 96                  | K/V chunk size (tile along seq dim) |
| `lq`      | 64              | 128                 | Q sequence length |
| `dk`      | 64              | 64                  | Key/head dimension |
| `dv`      | 64              | 64                  | Value/head dimension |

**Derived values** (using Makefile defaults: `lk=12288, lq=64`):
- `tile_size_q = lq = 64` (Q is *not* partitioned; each cascade column sees the full Q)
- `num_chunks = lk / lkp = 128` total K/V chunks
- `num_cascade_stages = 4`
- `chunks_per_stage = 128 / 4 = 32` chunks processed per cascade stage

**AIE2P matmul intrinsic**: `8×8×8` (vs AIE2's `4×8×4`)

---

## 3. Spatial Layout — 12 Cores on 3 Columns × 4 Rows

The design uses **3 herds** of 4 tiles each, arranged in a `[1, 4]` grid per herd.
The AIR compiler places them on a 3-column, 4-row compute tile array:

```
              Col 0          Col 1          Col 2
           (herd_0)       (herd_2)       (herd_1)
         ┌────────────┬────────────┬────────────┐
  Row 5  │  tile(0,5)  │  tile(1,5)  │  tile(2,5)  │  ← cascade stage 3 (last)
         │  Q·K matmul │  G·V matmul │  softmax    │
         ├────────────┼────────────┼────────────┤
  Row 4  │  tile(0,4)  │  tile(1,4)  │  tile(2,4)  │  ← cascade stage 2
         │  Q·K matmul │  G·V matmul │  softmax    │
         ├────────────┼────────────┼────────────┤
  Row 3  │  tile(0,3)  │  tile(1,3)  │  tile(2,3)  │  ← cascade stage 1
         │  Q·K matmul │  G·V matmul │  softmax    │
         ├────────────┼────────────┼────────────┤
  Row 2  │  tile(0,2)  │  tile(1,2)  │  tile(2,2)  │  ← cascade stage 0 (first — outputs result)
         │  Q·K matmul │  G·V matmul │  softmax    │
         └────────────┴────────────┴────────────┘

  Row 1     mem_tile(0,1) … mem_tile(7,1)   ← Mem tiles (L2)
  Row 0     shim_noc(0,0) … shim_noc(7,0)  ← Shim tiles (L3 DMA)
```

### Core Roles

| Herd | Column | Role | Key Operation |
|------|--------|------|---------------|
| `herd_0` | col 0 (tiles 0,2–0,5) | **Score computation** | `G = zero; G = Q·K` (matmul_a_b_bf16) |
| `herd_2` | col 1 (tiles 1,2–1,5) | **Value-weighted matmul** | `result = G·V` (matmul_g_b_bf16) |
| `herd_1` | col 2 (tiles 2,2–2,5) | **Softmax + accumulation + cascade merge** | max, exp, rescale, sum, cascade send/recv |

---

## 4. Cascade Pipeline (Column 2)

The 4 softmax tiles in column 2 form a **cascade chain** flowing from high row to low row:

```
tile(2,5) ──cascade──> tile(2,4) ──cascade──> tile(2,3) ──cascade──> tile(2,2)
  stage 3                stage 2                stage 1                stage 0
  (last chunk set)                                                    (first — produces output)
```

Each cascade stage processes `chunks_per_stage = 32` K/V chunks independently, maintaining
its own local `(Gp, up, sp)` state. After the main loop, the cascade reduction merges
partial results:

- **Stage 3** (tile 2,5): `put_cascade(Gp, up, sp)` — sends its partials downstream
- **Stages 2, 1** (tiles 2,4, 2,3): `get_cascade` → merge with local partials → `put_cascade`
  - Merge = rescale both sides to common max, add weighted sums
- **Stage 0** (tile 2,2): `get_cascade` → final merge → `div_gp_sp` → output to L2/L3

### Cascade Data Sent Per Stage

Data is transferred via `aie.put_cascade` / `aie.get_cascade` in 32-element bf16 vectors:
- **Gp** (64×64 bf16): 4096 elements → 128 cascade transfers
- **up** (64×1 bf16): 64 elements → 2 cascade transfers
- **sp** (64×1 bf16): 64 elements → 2 cascade transfers

Total per direction: **132 cascade words** (~8.25 KiB)

---

## 5. Data Movement Schedule

### 5.1 Memory Hierarchy

```
L3 (DDR/host)
  │
  ├── Q [lq × dk]         = [64 × 64] bf16 = 8 KiB
  ├── K [dk × lk]         = [64 × 12288] bf16 = 1.5 MiB
  ├── V [lk × dv]         = [12288 × 64] bf16 = 1.5 MiB
  └── output [lq × dv]    = [64 × 64] bf16 = 8 KiB
  │
  ▼ L3→L2 channels
L2 (Mem tiles, row 1)
  │
  ├── Q buffers:  4× [64 × 64] bf16 in mem_tiles 0–3   (one per cascade stage)
  ├── K buffers:  4× [64 × 96] bf16 in mem_tiles 0–3   (double-buffered per stage)
  ├── V buffers:  4× [96 × 64] bf16 in mem_tiles 5–7,0 (double-buffered per stage)
  └── Output buf: 1× [64 × 64] bf16 in mem_tile 4
  │
  ▼ L2→L1 channels
L1 (Compute tile local memory)
  │
  └── Per-tile working buffers (see §5.4)
```

### 5.2 L3 → L2 Transfers

#### Q matrix (one-time, before main loop)
- `L3ToL2Chan1[0, i]` for `i ∈ {0,1,2,3}`: Each stage gets a copy of Q (since `lq = tile_size_q`, the "tile" is the full Q matrix).
- Transfer: `[64, 64]` contiguous from `arg0` offset `[i*64, 0]`
  - **Note**: When `lq = 64` and `num_cascade_stages = 4`, the Q partitioning via `affine_map<()[s0] -> (s0 * 64)>` creates 4 copies starting at rows `{0, 64, 128, 192}`. However the built MLIR (with `lq=64`, matching Makefile) shows all 4 Q copies are identical `[64×64]` blocks — each cascade stage operates on the *same* Q, not a partition.

#### K matrix (streamed, per iteration)
- `L3ToL2Chan1[0, i]` for `i ∈ {0,1,2,3}`: Provides K chunks to each stage.
- Per-stage transfer pattern: `sizes=[32, 64, 96], strides=[384, 12288, 1]`, starting at column offset `i * 96`.
- This transfers 32 chunks of [64 × 96] bf16, interleaved across the 4 stages with stride-based gathering.
- Layout: K is `[dk × lk] = [64 × 12288]`, and the DMA gathers columns `[i*96 .. (i+1)*96)` for all 32 chunk iterations.

#### V matrix (streamed, per iteration)
- `L3ToL2Chan3[0, i]`: Provides V chunks similarly.
- Per-stage: `sizes=[32, 96, 64], strides=[24576, 64, 1]`, starting at row offset `i * 96`.
- Layout: V is `[lk × dv] = [12288 × 64]`, contiguous rows.

### 5.3 L2 → L1 Tiling Transforms

Data is retiled for the AIE2P 8×8×8 matmul intrinsic during L2→L1 transfer via DMA
BD (buffer descriptor) stride patterns:

#### Q (L2→L1, `L2ToL1Chan1`): `[64×64]` → `[8, 8, 8, 8]` with strides `[8, 512, 64, 1]`
- 4D tiling: `(dk/8, lq/8, 8, 8)` — rearranges for matmul A-input format
- Maps row-major `[lq, dk]` to block layout `[K_blocks, M_blocks, m, k]`

#### K (L2→L1, `L2ToL1Chan2`): `[64×96]` → `[12, 8, 8, 8]` with strides `[8, 768, 96, 1]`
- 4D tiling: `(lkp/8, dk/8, 8, 8)` — rearranges for matmul B-input format
- Maps `[dk, lkp]` to `[N_blocks, K_blocks, k, n]`

#### V (L2→L1, `L2ToL1Chan3`): `[96×64]` → `[8, 12, 8, 8]` with strides `[8, 512, 64, 1]`
- 4D tiling: `(dv/8, lkp/8, 8, 8)` — rearranges for matmul B-input format
- Maps `[lkp, dv]` to `[N_blocks, K_blocks, k, n]`

### 5.4 L1 ↔ L1 Channels (Inter-tile Communication)

Within each cascade stage, the three cores communicate via DMA-backed channels:

```
herd_0 (col 0)           herd_1 (col 2)           herd_2 (col 1)
   Q·K matmul              softmax core              G·V matmul
       │                       │                       │
       │  L1ToL1Chan1          │                       │
       ├───── G ──────────────>│                       │
       │   [64, 12, 8]         │  L1ToL1Chan2          │
       │   strides [8,512,1]   ├───── G_copy ─────────>│
       │                       │   [12, 64, 8]         │
       │                       │   strides [8,96,1]    │
       │                       │                       │
       │                       │  L1ToL1Chan3          │
       │                       │<───── G·V result ─────┤
       │                       │   [64, 8, 8]          │
       │                       │   strides [8,512,1]   │
```

**L1ToL1Chan1** (herd_0 → herd_1): G = Q·K result
- Shape `memref<6144xbf16>` (flat `64×96`)
- DMA BD retiling on send: `sizes=[64, 12, 8], strides=[8, 512, 1]` — transposes from matmul output layout to softmax input layout

**L1ToL1Chan2** (herd_1 → herd_2): softmax(G) for value multiplication
- Shape `memref<6144xbf16>` (flat `64×96`)
- DMA BD retiling on send: `sizes=[12, 64, 8], strides=[8, 96, 1]` — transposes to matmul A-input layout for G·V

**L1ToL1Chan3** (herd_2 → herd_1): G·V partial result
- Shape `memref<64×64xbf16>`
- DMA BD retiling on send: `sizes=[64, 8, 8], strides=[8, 512, 1]` — same tile-transpose pattern

### 5.5 Physical Routing (aie.flow)

From the lowered `aie.air.mlir`:

```
# L3 → L2 (K matrix)
shim_noc(0,0) DMA:0 → mem_tile(0,1) DMA:0    # K chunks for stage 0
shim_noc(1,0) DMA:0 → mem_tile(1,1) DMA:0    # K chunks for stage 1
shim_noc(2,0) DMA:0 → mem_tile(2,1) DMA:0    # K chunks for stage 2
shim_noc(3,0) DMA:0 → mem_tile(3,1) DMA:0    # K chunks for stage 3

# L3 → L2 (V matrix)
shim_noc(5,0) DMA:0 → mem_tile(5,1) DMA:0    # V chunks for stage 0
shim_noc(6,0) DMA:0 → mem_tile(6,1) DMA:0    # V chunks for stage 1
shim_noc(7,0) DMA:0 → mem_tile(7,1) DMA:0    # V chunks for stage 2

# L3 → L2 (Q + V stage 3)
shim_noc(0,0) DMA:1 → mem_tile(0,1) DMA:1    # Q for stage 0 + V for stage 3

# L2 → L1 (Q to herd_0)
mem_tile(0,1) DMA:0 → tile(0,2) DMA:0        # stage 0
mem_tile(1,1) DMA:0 → tile(0,3) DMA:0        # stage 1
mem_tile(2,1) DMA:0 → tile(0,4) DMA:0        # stage 2
mem_tile(3,1) DMA:0 → tile(0,5) DMA:0        # stage 3

# L2 → L1 (K to herd_0)
mem_tile(0,1) DMA:1 → tile(0,2) DMA:1
mem_tile(1,1) DMA:1 → tile(0,3) DMA:1
mem_tile(2,1) DMA:1 → tile(0,4) DMA:1
mem_tile(3,1) DMA:1 → tile(0,5) DMA:1

# L2 → L1 (V to herd_2)
mem_tile(5,1) DMA:0 → tile(1,2) DMA:0
mem_tile(6,1) DMA:0 → tile(1,3) DMA:0
mem_tile(7,1) DMA:0 → tile(1,4) DMA:0
mem_tile(0,1) DMA:2 → tile(1,5) DMA:0

# L1 ↔ L1 (inter-core within each stage)
tile(0,N) DMA:0 → tile(2,N) DMA:0            # G = Q·K result (herd_0 → herd_1)
tile(2,N) DMA:0 → tile(1,N) DMA:1            # softmax(G) copy (herd_1 → herd_2)
tile(1,N) DMA:0 → tile(2,N) DMA:1            # G·V result (herd_2 → herd_1)

# Cascade chain (herd_1 only)
tile(2,5) ──cascade──> tile(2,4) ──cascade──> tile(2,3) ──cascade──> tile(2,2)

# Output path
tile(2,2) DMA:0 → mem_tile(4,1) DMA:0        # L1 → L2
mem_tile(4,1) DMA:0 → shim_noc(4,0) DMA:0    # L2 → L3
```

---

## 6. Per-Core Buffer Inventory (L1)

### herd_0 (Q·K matmul) — col 0, each tile
| Buffer | Shape | Size (bf16) | Purpose |
|--------|-------|-------------|---------|
| Q_tile | 64×64 | 4096 (8 KiB) | Q input (loaded once) |
| K_chunk | 64×96 | 6144 (12 KiB) | K chunk (streamed per iteration) |
| G | 6144 flat | 6144 (12 KiB) | Q·K output (sent to herd_1) |
| **Total** | | **~32 KiB** | |

### herd_1 (softmax + cascade) — col 2, each tile
| Buffer | Shape | Size (bf16) | Purpose |
|--------|-------|-------------|---------|
| Gp | 64×64 | 4096 (8 KiB) | Running weighted-value accumulator |
| up | 64×1 | 64 (128 B) | Running row-max |
| sp | 64×1 | 64 (128 B) | Running row-sum |
| G_in | 6144 flat | 6144 (12 KiB) | Received Q·K scores |
| G_copy | 6144 flat | 6144 (12 KiB) | Copy for send to herd_2 |
| u_local | 64×1 | 64 (128 B) | Local max scratch |
| s_local | 64×1 | 64 (128 B) | Local sum scratch |
| r_local | 64×1 | 64 (128 B) | Rescale factor scratch |
| GV_result | 64×64 | 4096 (8 KiB) | Received G·V from herd_2 |
| *cascade buffers* | various | ~8.5 KiB | Gp_cascade, up_cascade, sp_cascade, scratch |
| **Total** | | **~49 KiB** | (heaviest tile) |

### herd_2 (G·V matmul) — col 1, each tile
| Buffer | Shape | Size (bf16) | Purpose |
|--------|-------|-------------|---------|
| G_in | 6144 flat | 6144 (12 KiB) | Received softmax(G) |
| V_chunk | 64×96 | 6144 (12 KiB) | V chunk (streamed per iteration) |
| result | 64×64 | 4096 (8 KiB) | G·V output (sent to herd_1) |
| **Total** | | **~32 KiB** | |

---

## 7. Kernel Functions (attn_aie2p.cc)

All kernels operate on bf16 data with **row-major layout** in L1.

| Function | Signature | Description |
|----------|-----------|-------------|
| `matmul_a_b_bf16` | `(Q, K, G)` | G += Q·K using 8×8×8 mmul, 2×2 tile expansion |
| `matmul_g_b_bf16` | `(G, V, out)` | out += G·V using same matmul template |
| `zero_fill_gp_bf16` | `(buf)` | Zero-fill 64×64 |
| `zero_fill_sp_bf16` | `(buf)` | Zero-fill 64×1 |
| `zero_fill_g_bf16` | `(buf)` | Zero-fill 64×96 (flat 6144) |
| `neg_inf_fill_up_bf16` | `(buf)` | Fill 64×1 with -inf |
| `max_g_bf16` | `(G, u)` | Row-wise max of G → u |
| `maximum_up_u_bf16` | `(up, u)` | Element-wise max(up, u) → u |
| `exp_g_minus_u` | `(u, G)` | G ← exp(G - u) in-place |
| `exp_up_minus_u` | `(up, u, r)` | r = exp(up - u) |
| `mul_r_gp` | `(r, Gp)` | Gp ← Gp * r (broadcast per row) |
| `sum_g` | `(G, s)` | Row-wise sum of G → s |
| `accum_sp_r_s` | `(sp, r, s)` | s += sp * r |
| `vector_copy_32elems` | `(off, in, out)` | Copy 64 bf16 elements (lq×1) |
| `vector_copy_32x96elems` | `(off, in, out)` | Copy 6144 bf16 elements (lq×lkp) |
| `vector_accum_32x64elems` | `(in, out)` | out += in for 4096 bf16 elements |
| `div_gp_sp` | `(sp, Gp)` | Gp ← Gp / sp (broadcast per row) |
| `add_gp_g` | `(gp, g)` | g += gp element-wise |

### Exp implementation (AIE2P)
Uses `exp2`-based approach: `exp(x) = 2^(x * log2(e))` via `aie::exp2<bfloat16>()`.
No LUT required — differs from AIE2 which uses LUT-based exp.

### Matmul template
Uses a **2×2 mmul unrolling** with the `aie::mmul<8,8,8,bf16,bf16,accauto>` intrinsic.
The outer loop iterates over M in steps of 2 (16 rows per outer iteration),
and the inner loop over N in steps of 2 (16 columns). The K dimension is fully
accumulated inside the innermost loop.

---

## 8. Execution Schedule (Per Iteration of Main Loop)

Within each K/V chunk iteration (32 iterations per stage), the three herds execute
concurrently:

```
Time ──────────────────────────────────────────────────────>

herd_0:  [zero G] [get K] [matmul Q·K] [put G via L1ToL1Chan1]
                                              │
herd_1:                        [get G from L1ToL1Chan1]
                               [max_g] [maximum_up_u] [exp_g_minus_u]
                               [exp_up_minus_u] [mul_r_gp]
                               [copy G → G_copy]
                               [put G_copy via L1ToL1Chan2] ──────────────────┐
                                              │                               │
                               [get GV from L1ToL1Chan3] ◄───────────────┐    │
                               [accum GV into Gp]                        │    │
                               [sum_g] [accum_sp_r_s]                    │    │
                               [copy s→sp, u→up]                         │    │
                                                                         │    │
herd_2:                                         [get G_copy from Chan2] ◄┘    │
                                                [zero result]                 │
                                                [get V]                       │
                                                [matmul G·V]                  │
                                                [put result via L1ToL1Chan3]──┘
```

After all 32 iterations, the cascade reduction runs **sequentially** from stage 3→0.

---

## 9. Key Design Decisions for IRON+Pythoc Port

### 9.1 Architecture Mapping

| AIR concept | IRON equivalent |
|-------------|-----------------|
| `air.launch` | Top-level program / npu_dma_memcpy_nd sequence |
| `air.segment` | Column group / device config |
| `air.herd [1,4]` | `@tile` workers placed on specific tiles |
| `air.channel` | `object_fifo` between tiles |
| `air.channel @cascade` | `aie.cascade_flow` / cascade object_fifo |
| External C kernel functions | `@kernel` functions linked via `link_with` |

### 9.2 Tile Placement

The IRON implementation should explicitly place tiles:

```
# Column 0: Q·K matmul (herd_0)
qk_tiles = [tile(0,2), tile(0,3), tile(0,4), tile(0,5)]

# Column 1: G·V matmul (herd_2)
gv_tiles = [tile(1,2), tile(1,3), tile(1,4), tile(1,5)]

# Column 2: Softmax + cascade (herd_1)
sf_tiles = [tile(2,2), tile(2,3), tile(2,4), tile(2,5)]
```

### 9.3 Object FIFOs Needed

| FIFO | Producer → Consumer | Shape | Notes |
|------|---------------------|-------|-------|
| `of_q[i]` | shim → qk_tile[i] | 64×64 bf16 | One-shot, retiled 4D |
| `of_k[i]` | shim → qk_tile[i] | 64×96 bf16 | Streamed ×32, retiled 4D |
| `of_v[i]` | shim → gv_tile[i] | 96×64 bf16 | Streamed ×32, retiled 4D |
| `of_g[i]` | qk_tile[i] → sf_tile[i] | 6144 bf16 | Per-iter, retiled on both ends |
| `of_gcopy[i]` | sf_tile[i] → gv_tile[i] | 6144 bf16 | Per-iter, retiled on send |
| `of_gv[i]` | gv_tile[i] → sf_tile[i] | 64×64 bf16 | Per-iter, retiled on send |
| `of_out` | sf_tile[0] → shim | 64×64 bf16 | Single output transfer |
| `cascade_fifo` | sf_tile[3]→[2]→[1]→[0] | cascade | Gp+up+sp |

### 9.4 DMA Tiling Transforms (BD Patterns)

These are the critical 4D tiling transforms that must be reproduced:

**Q (L2→L1)**: `[64×64]` → `[dk/8, lq/8, 8, 8]` = `[8, 8, 8, 8]`
- Strides: `[8, 512, 64, 1]`

**K (L2→L1)**: `[64×96]` → `[lkp/8, dk/8, 8, 8]` = `[12, 8, 8, 8]`
- Strides: `[8, 768, 96, 1]`

**V (L2→L1)**: `[96×64]` → `[dv/8, lkp/8, 8, 8]` = `[8, 12, 8, 8]`
- Strides: `[8, 512, 64, 1]`

**G (L1→L1, herd_0→herd_1)**: flat `6144` → `[64, 12, 8]`
- Strides: `[8, 512, 1]`

**G_copy (L1→L1, herd_1→herd_2)**: flat `6144` → `[12, 64, 8]`
- Strides: `[8, 96, 1]`

**GV result (L1→L1, herd_2→herd_1)**: `[64×64]` → `[64, 8, 8]`
- Strides: `[8, 512, 1]`

### 9.5 Cascade Merge Logic

The cascade merge at stages 2, 1, and 0 is identical — the pseudocode is:

```python
# Receive from upstream cascade
Gp_A, up_A, sp_A = cascade_recv()

# Save local max before overwrite
up_B_saved = copy(up_local)

# Merge maxima
up_merged = max(up_A, up_local)   # overwrites up_local

# Compute rescale factors
r_A = exp(up_A - up_merged)
r_B = exp(up_B_saved - up_merged)

# Rescale both accumulators
Gp_A *= r_A
Gp_local *= r_B

# Merge
Gp_merged = Gp_local + Gp_A      # stored in Gp_A for cascade send

# Merge sums
sp_merged = sp_A * r_A + sp_local * r_B
```

At **stage 0 only**: after merge, apply `Gp_merged /= sp_merged` and output to L2.

### 9.6 K/V Striding from L3

K chunks for different stages are **interleaved** in the L3 K matrix:
- Stage `i` gets columns `[i*96 .. (i+1)*96)` for each chunk
- With 4 stages, chunk `j` maps to `K[:, j*96*4 + i*96 : j*96*4 + (i+1)*96]`
- The DMA BD stride pattern handles this gathering.

V chunks are similarly interleaved along the row dimension.

---

## 10. Summary Table: Tile-to-Buffer-to-Channel Mapping

| Stage | QK tile (col 0) | SF tile (col 2) | GV tile (col 1) |
|-------|-----------------|-----------------|-----------------|
| 0 | tile(0,2): buf2(Q), buf1(K), buf0(G) | tile(2,2): buf39(Gp), buf37(up), buf38(sp), buf33(G_in), buf32(G_copy), buf31(GV), buf29(out) | tile(1,2): buf22(G_in), buf23(V), buf21(result) |
| 1 | tile(0,3): buf5(Q), buf4(K), buf3(G) | tile(2,3): buf55(Gp), buf53(up), buf54(sp), buf49(G_in), buf48(G_copy), buf47(GV), buf45(cascade_Gp) | tile(1,3): same pattern |
| 2 | tile(0,4): buf8(Q), buf7(K), buf6(G) | tile(2,4): similar | tile(1,4): similar |
| 3 | tile(0,5): buf11(Q), buf10(K), buf9(G) | tile(2,5): buf80(Gp), buf78(up), buf79(sp), buf74(G_in), buf73(G_copy), buf72(GV) | tile(1,5): similar |

---

## 11. Notes for Pythoc Implementation

1. **Kernel reuse**: The same C++ kernels (`attn_aie2p.cc`) can be compiled and linked unchanged. They use row-major layout matching the tiled DMA patterns.

2. **No mask support in built design**: Although `input_m` (mask) is passed as an argument, the Makefile-default build uses `lq=64` and the mask appears to be zeros. The kernel `matmul_a_b_bf16` does `G += Q·K` (accumulating into pre-zeroed G), so mask addition would need a separate kernel or pre-add.

3. **Wrap-around loop**: The generated AIE MLIR uses `cf.br ^bb1` for an infinite outer loop (runtime-controlled repeat), which is the standard IRON pattern with `while True` in the worker.

4. **Lock patterns**: Each DMA channel uses a producer/consumer lock pair with `init=1` (producer) and `init=0` (consumer) for single-buffering. Double-buffering could improve performance.

5. **Single Q tile**: With `lq=64`, there's no partitioning of Q across rows. Each cascade stage processes all 64 Q rows against its 32 K/V chunks. The `num_q_tiles=4` parameter in the Python defaults is not actually used in the tiling — `tile_size_q = lq` always.

# AWQ Matvec Dataflow

Summary of the AWQ (Activation-aware Weight Quantization) matrix-vector design:
how weights are packed on the host, how the cores unpack them, and the
activation distribution pattern.

## 1. Weight packing (host side)

Packing uses the **combined-row ABI** — for each output row, the quantized
weights and their per-group scale/zero params are concatenated into a single
`uint8` buffer.

`awq_combined_weight()` (`llama32_1b_awq_runtime.py:113-129`) builds a row-major
`uint8[M, K/2 + 4*groups]` where each row is:

```
[ qweight : K/2 bytes ] [ params : 4*groups bytes ]
```

- **qweight** — `uint4` nibbles packed two-per-byte: low nibble = even K, high
  nibble = odd K (AWQ packing convention). `K/2` bytes per row.
- **params** — per-group `[scale, zero]` bf16 pairs, bitcast to uint8
  (`p.view(np.uint8)`), so `2*groups` bf16 = `4*groups` bytes appended after the
  qweight section.

Concrete row widths (`GROUP_SIZE = 128`):

| K | qweight | params | row_bytes |
|---|---------|--------|-----------|
| 2048 (og/gg/ug) | 1024 | 64 (16 groups) | **1088** |
| 8192 (down-proj) | 4096 | 256 (64 groups) | **4352** |

It is a per-row concatenation of `[packed-nibbles | interleaved scale/zero
pairs]`, not three separate buffers. One DMA stream carries everything a row
needs.

## 2. Core unpacking / dequant

The kernel (`kernels/awq_mv.py:108`, `awq_matvec_vectorized_u4_bf16`) walks each
row in `combined_in`, splitting it back into the qweight pointer and the params
pointer:

```python
q_row = row_base                    # nibbles
p_row = row_base + packed_per_row   # scale/zero pairs (k//2 offset)
```

Per group (128 K-elements = 64 packed bytes):

1. **Load + nibble-unpack** — `load_v(...,64)` then `unpack_I1024_I8_I4` expands
   64 bytes → 128 `u8` nibbles in K-natural order `[byte0_lo, byte0_hi, ...]`
   (`awq_mv.py:199-208`).
2. **uint4 → bf16 via Fix2Float magic-number trick** (`awq_mv.py:215-229`),
   because `uitofp <N x bf16>` and vector `fadd/fsub` don't legalize on AIE2P
   GISel:
   - zero-extend nibbles to i32 (UPS chain),
   - integer-add magic `0x4b010000` (`ACC2048_add_conf`),
   - bitcast to accfloat,
   - subtract the magic back out as bf16 via an MSC (`ACC2048_accfloat_sub_conf`,
     conf=60),
   - `v32accfloat_to_v32bf16` → final bf16.
3. **Dequant fused into the MAC** (`awq_mv.py:186, 231-243`). Instead of
   `(w - zero)*scale`, it precomputes `zs = zero*scale` and does
   `acc += x*(w*s) - x*zs`:
   - `w_scaled = w_bf * scale` (one bf16 mul),
   - `I1024_..._bf_mac_conf(x, w_scaled, acc)` then
     `I1024_..._bf_msc_conf(x, zs_v, acc)`.

Accumulation is a single **64-lane f32 accumulator** per output row; two 64-lane
MAC/MSC pairs cover the 128-element group. At the end,
`reduce_add_reassoc(acc)` collapses to a scalar and stores `bf16`
(`awq_mv.py:276-278`).

The `awq_mv_k8192.py` variant is identical math, just 64 groups/row instead
of 16.

## 3. Activation pattern

The pattern is **weight-partitioned, activation-broadcast**:

- **Activation X is NOT partitioned.** The full input vector (`bf16[2048]`, or
  `bf16[8192]` for down-proj) is broadcast from the shim to *all 8 compute
  cores* — a single MM2S channel fans out to `{tile(c,2) | c in 0..7}`. Every
  core holds the whole X in L1.
- **Weights ARE partitioned by output row** across the 8 columns. Each column
  owns a contiguous slice of output rows and streams only its weight rows
  DDR → memtile (L2) → core (L1).
- **No real K-tiling** in the og/gg/ug path: `K_TILE = M_TILE = 8`, so the
  K-loop is a single iteration — all of K=2048 is consumed in one kernel call.
  K=8192 down-proj uses `M_TILE = 2`.

**Output row distribution (e.g. OG, 2048 rows):** 2 outer iters × 1024 rows,
128 rows/column/outer, 16 kernel calls/column (M_TILE=8 rows each). Output
offset per col = `outer*1024 + col*8`.

### Picture

```
DDR combined[M, row_bytes]            x[K]                 (broadcast)
   │ (row-sliced by column)            │
   ▼                                   ▼
shim(c,0) ──► memtile(c,1) ──► core(c,2):  [q | scale,zero] + full x[K]
                                         unpack u4->bf16, fused MAC/MSC
                                         64-lane f32 acc -> reduce -> y[8]
                                   ▼
                              y rows gathered back to DDR
   (8 columns in parallel, each owning a disjoint band of output rows)
```

Net: each row is a self-contained `[nibbles | scale/zero]` packet, the
activation is replicated everywhere, and the 8 columns split the output
dimension — so the design is DMA-streaming weights with a fully-resident
activation, which is why it is DMA-bandwidth bound on the weight stream.

## Key files

- `llama32_1b_awq_runtime.py:113-129` — `awq_combined_weight()` host packing
- `kernels/awq_mv.py` — K=2048 core kernel (og/gg/ug)
- `kernels/awq_mv_k8192.py` — K=8192 core kernel (down-proj)
- `builders/awq_matvec.py` — standalone GEMV builder (single tile, packet flows)
- `builders/o_gemv_ffn_awq.py` — fused O+FFN builder (8-column herd, memtiles)

## Reference: buffer shapes & layout

| Parameter | Value | Notes |
|-----------|-------|-------|
| EMB_DIM | 2048 | model hidden dimension |
| HIDDEN_DIM | 8192 | FFN intermediate dimension |
| GROUP_SIZE | 128 | AWQ quant group size (baked) |
| N_COLS | 8 | compute columns in herd |
| K_TILE / M_TILE (K=2048) | 8 / 8 | K-loop is a single iteration |
| K_TILE / M_TILE (K=8192) | 1 / 2 | down-proj |
| row_bytes (K=2048) | 1088 | 1024 qweight + 64 params |
| row_bytes (K=8192) | 4352 | 4096 qweight + 256 params |
| L1 activation (K=2048 / K=8192) | 4 KB / 16 KB | bf16[K], broadcast (not partitioned) |

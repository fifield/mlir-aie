# Flash Attention — IRON + PythoC Implementation Plan

> **Goal**: Port the AIR-based `dataflow_based/attn.py` flash attention to IRON + PythoC,
> targeting the same AIE2P 12-core architecture described in `aie2p_flash_attention_analysis.md`.

---

## Phase 1: IRON Topology + C++ Kernels (Option 2)

Build the full IRON program (ObjectFifos, Workers, Runtime sequence, cascade wiring)
using the existing `attn_aie2p.cc` C++ kernels via `ExternalFunction` / `Kernel`.
This validates the topology and data movement before touching kernel code.

### 1.1 — Scaffold and constants

Create `flash_attention/flash_attention.py` with:
- Problem-size constants (`lk`, `lkp`, `lq`, `dk`, `dv`, `num_cascade_stages`,
  `chunks_per_stage`) matching the Makefile defaults (`lk=12288, lq=64, lkp=96, dk=64, dv=64`).
- Derived tiling constants for the 8×8×8 mmul (`R=S=T=8`).
- CLI argument parsing (mirror `attn.py`'s args + `--compile-only`, `--work-dir`).
- NumPy reference implementation (lines 780-801 of `attn.py`) for golden comparison.

### 1.2 — Tile placement

Explicitly place 12 compute tiles + mem tiles following the analysis:

```
Col 0 (QK matmul):  tiles (0,2), (0,3), (0,4), (0,5)
Col 1 (GV matmul):  tiles (1,2), (1,3), (1,4), (1,5)
Col 2 (softmax):    tiles (2,2), (2,3), (2,4), (2,5)
```

Use `Tile(col, row)` placement for each Worker. Mem tiles at row 1 are implicitly
used by ObjectFifo placement.

### 1.3 — External kernel declarations

Declare all 17 C++ kernel functions from `attn_aie2p.cc` as IRON `ExternalFunction`
(or `Kernel` if pre-compiled `.o` is supplied). Each needs its memref type signature:

| Function | Args (L1 types) |
|----------|-----------------|
| `matmul_a_b_bf16` | `(lq×dk bf16, dk×lkp bf16, lq*lkp bf16)` |
| `matmul_g_b_bf16` | `(lq*lkp bf16, dk×lkp bf16, lq×dk bf16)` |
| `zero_fill_gp_bf16` | `(lq×dk bf16,)` |
| `zero_fill_sp_bf16` | `(lq×1 bf16,)` |
| `zero_fill_g_bf16` | `(lq*lkp bf16,)` |
| `neg_inf_fill_up_bf16` | `(lq×1 bf16,)` |
| `max_g_bf16` | `(lq*lkp bf16, lq×1 bf16)` |
| `maximum_up_u_bf16` | `(lq×1 bf16, lq×1 bf16)` |
| `exp_g_minus_u` | `(lq×1 bf16, lq*lkp bf16)` |
| `exp_up_minus_u` | `(lq×1 bf16, lq×1 bf16, lq×1 bf16)` |
| `mul_r_gp` | `(lq×1 bf16, lq×dk bf16)` |
| `sum_g` | `(lq*lkp bf16, lq×1 bf16)` |
| `accum_sp_r_s` | `(lq×1 bf16, lq×1 bf16, lq×1 bf16)` |
| `vector_copy_32elems` | `(i32, lq×1 bf16, lq×1 bf16)` |
| `vector_copy_32x96elems` | `(i32, lq*lkp bf16, lq*lkp bf16)` |
| `vector_accum_32x64elems` | `(lq×dk bf16, lq×dk bf16)` |
| `div_gp_sp` | `(lq×1 bf16, lq×dk bf16)` |
| `add_gp_g` | `(lq×dk bf16, lq×dk bf16)` |

Source: `attn_aie2p.cc`, compiled with `-Dlqp=64 -Dlkp=96 -Ddk=64 -Ddv=64`.

### 1.4 — ObjectFifo topology (L3 → L2 → L1)

For each cascade stage `i ∈ {0,1,2,3}`:

**Q (one-shot, before main loop)**:
- `of_q[i]`: L3 → mem_tile → QK tile.
  - L2→L1 `dims_to_stream`: `[(dk/8, 8), (lq/8, dk*8), (8, dk), (8, 1)]`
    producing `[8, 8, 8, 8]` with strides `[8, 512, 64, 1]`.
  - Depth 1 (loaded once).

**K (streamed, 32 iterations)**:
- `of_k[i]`: L3 → mem_tile → QK tile.
  - L3→L2: stride-gather pattern selecting stage `i`'s interleaved columns.
  - L2→L1 `dims_to_stream`: `[(lkp/8, 8), (dk/8, lkp*8), (8, lkp), (8, 1)]`
    producing `[12, 8, 8, 8]` with strides `[8, 768, 96, 1]`.
  - Depth 2 (double-buffered).

**V (streamed, 32 iterations)**:
- `of_v[i]`: L3 → mem_tile → GV tile.
  - L3→L2: stride-gather for stage `i`'s interleaved rows.
  - L2→L1 `dims_to_stream`: `[(dv/8, 8), (lkp/8, dv*8), (8, dv), (8, 1)]`
    producing `[8, 12, 8, 8]` with strides `[8, 512, 64, 1]`.
  - Depth 2 (double-buffered).

### 1.5 — ObjectFifo topology (L1 ↔ L1 inter-tile)

For each cascade stage `i`:

**G scores** (`of_g[i]`): QK tile[i] → SF tile[i]
- Type: flat `(lq*lkp,) bf16` = `(6144,) bf16`
- Producer `dims_to_stream`: `[(lq, 8), (lkp/8, lq*8), (8, 1)]`
  → `[64, 12, 8]` with strides `[8, 512, 1]`
- Depth 1.

**G copy for V matmul** (`of_gcopy[i]`): SF tile[i] → GV tile[i]
- Type: flat `(lq*lkp,) bf16` = `(6144,) bf16`
- Producer `dims_to_stream`: `[(lkp/8, 8), (lq, lkp), (8, 1)]`
  → `[12, 64, 8]` with strides `[8, 96, 1]`
- Depth 1.

**GV result** (`of_gv[i]`): GV tile[i] → SF tile[i]
- Type: `(lq, dv) bf16` = `(64, 64) bf16`
- Producer `dims_to_stream`: `[(lq, 8), (dv/8, lq*8), (8, 1)]`
  → `[64, 8, 8]` with strides `[8, 512, 1]`
- Depth 1.

### 1.6 — Output ObjectFifo

**Result** (`of_out`): SF tile[0] (stage 0, tile 2,2) → mem_tile → shim → L3.
- Type: `(lq, dv) bf16` = `(64, 64) bf16`
- Single transfer after cascade merge completes.

### 1.7 — Cascade wiring

Wire `cascade_flow` between the softmax tiles (column 2):
```
tile(2,5) → tile(2,4) → tile(2,3) → tile(2,2)
```

The cascade carries `Gp` (64×64), `up` (64×1), `sp` (64×1) — transferred
element-by-element using `put_cascade` / `get_cascade` in the worker bodies.

### 1.8 — Worker: QK matmul (herd_0, 4 instances)

One Worker per stage on col 0 tiles. Pseudo-code:

```python
def qk_worker(of_q_cons, of_k_cons, of_g_prod, matmul_a_b, zero_fill_g):
    q_tile = of_q_cons.acquire(1)        # load Q once
    for _ in range_(chunks_per_stage):
        g_tile = of_g_prod.acquire(1)
        zero_fill_g(g_tile)
        k_tile = of_k_cons.acquire(1)
        matmul_a_b(q_tile, k_tile, g_tile)
        of_k_cons.release(1)
        of_g_prod.release(1)
    of_q_cons.release(1)
```

### 1.9 — Worker: GV matmul (herd_2, 4 instances)

One Worker per stage on col 1 tiles. Pseudo-code:

```python
def gv_worker(of_gcopy_cons, of_v_cons, of_gv_prod, matmul_g_b, zero_fill_gp):
    for _ in range_(chunks_per_stage):
        gv_tile = of_gv_prod.acquire(1)
        zero_fill_gp(gv_tile)
        g_tile = of_gcopy_cons.acquire(1)
        v_tile = of_v_cons.acquire(1)
        matmul_g_b(g_tile, v_tile, gv_tile)
        of_v_cons.release(1)
        of_gcopy_cons.release(1)
        of_gv_prod.release(1)
```

### 1.10 — Worker: Softmax + cascade (herd_1, 4 instances)

This is the most complex worker. Each instance has a different role based on its
cascade position, but all share the same code path with conditionals.

**Main loop** (runs `chunks_per_stage` times):

```python
# Receive G from QK core
g_tile = of_g_cons.acquire(1)

# Online softmax update
u = max_g(g_tile)
maximum_up_u(up, u)
exp_g_minus_u(u, g_tile)
r = exp_up_minus_u(up, u)
mul_r_gp(r, Gp)

# Send G copy to GV core
g_copy = of_gcopy_prod.acquire(1)
vector_copy_32x96(g_tile, g_copy)
of_g_cons.release(1)
of_gcopy_prod.release(1)

# Receive GV result and accumulate
gv_tile = of_gv_cons.acquire(1)
vector_accum(gv_tile, Gp)
of_gv_cons.release(1)

# Update running sum
s = sum_g(g_tile)
accum_sp_r_s(sp, r, s)
sp = s; up = u
```

**Post-loop cascade reduction** (conditional on stage index):

- **Stage 3** (last): `put_cascade(Gp, up, sp)`
- **Stages 2, 1** (middle): `get_cascade(Gp_A, up_A, sp_A)` → merge → `put_cascade(merged)`
- **Stage 0** (first): `get_cascade(Gp_A, up_A, sp_A)` → merge → `div_gp_sp` → output

The cascade merge logic is identical for stages 2/1/0 (see analysis §9.5).
Stage 0 additionally normalizes and writes to `of_out`.

**Implementation note**: The stage index can be determined either by:
- Using 4 separate worker functions (simplest, matches the lowered AIE MLIR), or
- Passing the stage index as a worker argument and branching.

Recommend 4 separate worker functions since the post-loop code differs significantly.

### 1.11 — Runtime sequence

```python
rt = Runtime()
with rt.sequence(Q_ty, K_ty, V_ty, mask_ty, output_ty) as (Q, K, V, mask, out):
    rt.start(*all_workers)

    # Fill Q (one-shot to all 4 stages)
    for i in range(4):
        rt.fill(of_q[i].prod(), Q, tap=q_taps[i])

    # Stream K and V (32 chunks × 4 stages, interleaved)
    for i in range(4):
        rt.fill(of_k[i].prod(), K, tap=k_taps[i])
        rt.fill(of_v[i].prod(), V, tap=v_taps[i])

    # Drain output
    rt.drain(of_out.cons(), out, tap=out_tap, wait=True)
```

The K/V taps must use stride patterns that extract the interleaved chunks for each
stage (see analysis §5.2). Use `TensorTiler2D` or manual tap construction.

### 1.12 — Compile, link, and test

- Compile `attn_aie2p.cc` with Peano (`-Dlqp=64 -Dlkp=96 -Ddk=64 -Ddv=64`).
- Build IRON `Program`, resolve with `SequentialPlacer` (or explicit placement).
- Compile MLIR → xclbin.
- Run with XRT against NumPy golden.
- Compare generated `aie.mlir` against the reference `aie.air.mlir` for structural
  correctness (tile placement, flow routing, DMA BDs).

---

## Phase 2: Replace C++ Kernels with PythoC (Option 1)

Replace each `ExternalFunction` with a `PythocKernel` equivalent, one at a time,
testing after each replacement. Ordered from simplest → hardest.

### 2.1 — Fill/copy utilities (trivial, validate PythoC plumbing)

| Kernel | Complexity | Notes |
|--------|-----------|-------|
| `zero_fill_gp_bf16` | Trivial | `store_v` loop, zeros, `lq*dv` elements |
| `zero_fill_sp_bf16` | Trivial | Same pattern, `lq` elements |
| `zero_fill_g_bf16` | Trivial | Same pattern, `lq*lkp` elements |
| `neg_inf_fill_up_bf16` | Trivial | Broadcast `0xff80` (bf16 -inf), store loop |
| `vector_copy_32elems` | Trivial | `load_v`/`store_v` loop, `lq` bf16 elements |
| `vector_copy_32x96elems` | Trivial | Same, `lq*lkp` elements |

### 2.2 — Element-wise vector operations

| Kernel | Complexity | Notes |
|--------|-----------|-------|
| `maximum_up_u_bf16` | Simple | `max(up, u)` per 8-element vector |
| `mul_r_gp` | Simple | Row-broadcast multiply: `Gp[row,:] *= r[row]` |
| `add_gp_g` | Simple | Element-wise add, `lq*dv` elements |
| `vector_accum_32x64elems` | Simple | `out += in` with accfloat intermediate |
| `div_gp_sp` | Simple | Row-broadcast divide: `Gp[row,:] /= sp[row]` |

### 2.3 — Reduction operations

| Kernel | Complexity | Notes |
|--------|-----------|-------|
| `max_g_bf16` | Medium | Row-wise max over `lkp` columns → scalar per row. Needs `reduce_max`. |
| `sum_g` | Medium | Row-wise sum over `lkp` columns → scalar per row. Needs `reduce_add`. |
| `accum_sp_r_s` | Medium | Fused `s += sp * r`. Uses accfloat for precision. |

### 2.4 — Exponential

| Kernel | Complexity | Notes |
|--------|-----------|-------|
| `exp_g_minus_u` | Medium-Hard | `G ← exp(G - u)` in-place. Needs `exp2` intrinsic: `exp(x) = 2^(x * log2e)`. PythoC needs `aie::exp2<bf16>` binding. |
| `exp_up_minus_u` | Medium-Hard | `r = exp(up - u)`. Same exp2 approach. |

**PythoC prerequisite**: Ensure `aie.exp2` (or equivalent) is exposed. If not,
this is a PythoC intrinsic addition task.

### 2.5 — Matrix multiplication

| Kernel | Complexity | Notes |
|--------|-----------|-------|
| `matmul_a_b_bf16` | Hard | Q×K matmul, `lq×dk × dk×lkp`. Uses `aie::mmul<8,8,8>` with 2×2 unrolling. Can reuse the existing bf16_gemm PythoC kernel adapted for bf16→bf16 (no BFP16 emulation) or write a new simpler version using `aie::mmul` directly. |
| `matmul_g_b_bf16` | Hard | G×V matmul, `lq×lkp × lkp×dv`. Same template, different dimensions. |

**Key difference from bf16_gemm_multi_core**: The flash attention matmuls use
`aie::mmul<8,8,8,bf16,bf16,accauto>` with bf16 output (not BFP16 emulation with
f32 accum). This is a simpler matmul than the GEMM example — no `v64accfloat_to_v64bfp16ebs8`
conversion needed. If PythoC exposes `aie::mmul` or the underlying MAC intrinsic for
bf16×bf16→bf16, the kernel is straightforward.

### 2.6 — Cascade put/get

The cascade data transfer in the worker bodies uses `put_cascade` and `get_cascade`
to move vectors of 32×bf16 through the cascade interface. PythoC should expose these
as intrinsics. The transfer loops are simple:

```python
# put_cascade: send Gp (4096 bf16 = 128 × 32-element vectors)
for i in range_(0, 4096, 32):
    v = load_v(gp_buf + i, 32)
    put_cascade(v)
```

---

## Testing Strategy

Each phase/task is tested by running the full flash attention program and comparing
output against the NumPy golden. The tolerance is `rtol=1e-1` (bf16 precision).

- **Phase 1 complete**: All C++ kernels, IRON topology → functional test passes.
- **Phase 2.1**: Replace fill/copy kernels → functional test still passes.
- **Phase 2.2**: Replace element-wise kernels → test passes.
- **Phase 2.3**: Replace reduction kernels → test passes.
- **Phase 2.4**: Replace exp kernels → test passes.
- **Phase 2.5**: Replace matmul kernels → test passes.
- **Phase 2.6**: Cascade intrinsics in PythoC → test passes, no C++ dependency remains.

## File Layout

```
flash_attention/
├── aie2p_flash_attention_analysis.md    # Reference analysis (read-only)
├── flash_attention_plan.md              # This plan (read-only)
├── flash_attention.py                   # Main IRON program
├── attn_aie2p.cc                        # C++ kernels (copied, Phase 1)
├── zero.cc                              # Zero-fill helper (copied, Phase 1)
└── build/                               # Build artifacts
```

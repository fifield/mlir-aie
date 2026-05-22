# o_gemv_ffn: placed-IRON builder vs. cached AIR reference

This note tracks the structural diff between
`builders/o_gemv_ffn.build_o_gemv_ffn_module()` and the cached
AIR-stitched `reference_mlir/o_gemv_ffn.npu.air.mlir`.

## Bottom line

The two MLIR modules are **structurally equivalent on every aie/aiex
op count except `func.call`** (mine has +1 — see "Cosmetic diffs"
below). `aiecc` compiles the placed-IRON output in ~4.6 s (same as
the cached path), produces an ELF that passes the HF answer gate
(`A: The capital of France is Paris.`) and the snapshot regression
gate (identical decode token IDs to the cached baseline; logits corr,
K/V cache corr identical to 6 decimal places; NPU/CPU top-1 = 9226).

| Metric                          | placed-IRON | cached AIR |
| ------------------------------- | ----------- | ---------- |
| total MLIR lines                | 8,180       | 7,964      |
| `aie.device`                    | 9           | 9          |
| `aie.tile`                      | 146         | 146        |
| `aie.lock`                      | 470         | 470        |
| `aie.buffer`                    | 236         | 236        |
| `aie.flow`                      | 235         | 235        |
| `aie.shim_dma_allocation`       | 143         | 143        |
| `aie.mem`                       | 57          | 57         |
| `aie.memtile_dma`               | 32          | 32         |
| `aie.core`                      | 57          | 57         |
| `aie.runtime_sequence`          | 9           | 9          |
| `aiex.dma_configure_task_for`   | 517         | 517        |
| `aiex.dma_start_task`           | 517         | 517        |
| `aiex.dma_await_task`           | 233         | 233        |
| `aiex.dma_free_task`            | 284         | 284        |
| `aiex.configure`                | 8           | 8          |
| `aiex.run`                      | 8           | 8          |
| `aie.dma_bd`                    | 816         | 816        |
| `aie.use_lock`                  | 940         | 940        |
| `func.call`                     | 73          | 72         |
| HF answer-gate result           | PASS        | PASS       |
| snapshot decode token IDs       | identical   | -          |
| logits corr                     | 0.854103    | 0.854103   |
| K_cache corr min                | 0.882564    | 0.882564   |
| V_cache corr min                | 0.883117    | 0.883117   |
| npu_top1 / cpu_top1             | 9226 / 9226 | 9226 / 9226|
| decode tok/s (HF gate)          | 7.83        | ~7.55–7.9  |
| aiecc compile time              | 4.6 s       | 4.5 s      |

## Module layout

Eight segment devices + one dispatcher (firing order matches the
cached IR's `@o_gemv_ffn` runtime_sequence):

  * `@og_matvec_bf16_0`   — 8 compute tiles, O GEMV (2048×2048),
    `mv_pythoc.o`, 2 outer iters.
  * `@a1_eltwise_add_seg` — 8 compute tiles, inline `arith.addf`
    on 256-elt bf16 tiles (proj + x_residual → res1).
  * `@rm_rms_seg`         — 1 compute tile, RMSNorm on 2048-elt
    (res1 × ffn_norm_w → normed2). External `rms_norm_2048_bf16.o`
    (cached AIR keeps the body inline — see "Cosmetic diffs").
  * `@gg_matvec_bf16_0`   — 8 compute tiles, Gate GEMV (8192×2048),
    `mv_pythoc.o`, 8 outer iters.
  * `@ug_matvec_bf16_0`   — 8 compute tiles, Up GEMV (8192×2048),
    `mv_pythoc.o`, 8 outer iters.
  * `@sw_silu_mul_seg`    — 8 compute tiles, SwiGLU on 1024-elt
    per-tile chunks (`silu_and_mul_bf16.o`).
  * `@dg_matvec_bf16_0`   — 8 compute tiles, Down GEMV (2048×8192),
    `mv_k8192_pythoc.o`, 8 outer iters (256 rows per outer).
  * `@a2_eltwise_add_seg` — 8 compute tiles, inline `arith.addf`
    on 256-elt bf16 tiles (down + res1 → output).
  * (unnamed) dispatcher  — fires the 8 segments in topo order
    (`og → a1 → rm → gg → ug → sw → dg → a2`).

All 9 runtime sequences share the same 15-arg host signature; the
arg layout is dictated by the dispatcher device's
`aiex.runtime_sequence @o_gemv_ffn(...)` block in the cached IR.

## Cosmetic differences

These come purely from how MLIR's printer renders auto-generated SSA
names + which named attributes are emitted by aircc but not by our
builder.  None affect correctness or `aiecc` compilation.

1. **SSA value names** -- AIR's aircc emits explicit `sym_name`
   attributes like `buf3`, `lock_0_2_63`; our builder lets MLIR
   auto-name (`buffer_0_2`, `lock_0_2_63`). Same number of locks /
   buffers, different printed text.

2. **`task_id = 0 : i32` attribute on `aie.dma_bd`** -- present in
   the cached IR (aircc adds it during stitching), absent in our
   builder's output.  Optional metadata, not consumed by `aiecc`.

3. **Cores' `air.herd_*` attributes** -- e.g.
   `{air.herd_local_id = array<i64: 7, 0>, air.herd_name = "og_herd_0",
   air.herd_size = array<i64: 8, 1>, link_with = "mv_pythoc.o"}`.
   AIR-only metadata; the `link_with` is reattached by the
   `aie-assign-core-link-files` pass from each `external_func` decl's
   `link_with` attribute.

4. **`#loop_annotation = #llvm.loop_annotation<mustProgress = true>`**
   -- a top-level attr aircc attaches via its `LowerHerds` pass.

5. **`{dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}` attribute on
   each segment device** -- AIR emits it; we don't.  `aiecc` does
   not require it.

6. **Core body shape** -- AIR emits `cf.br ^bb1` infinite loops
   wrapping the scf.for; our builder uses
   `for _ in range_(_sys.maxsize)` which lowers to an
   `scf.for c0=0 to maxsize step c1`.  Semantically equivalent.

7. **rm_rms_seg: external_func vs inline body.**  The cached AIR
   reference inlines the RMSNorm math directly in the core body using
   vector.transfer_read / arith.mulf / vector.reduction.  Our builder
   reuses `kernel_builder.cache`'s already-built
   `rms_norm_2048_bf16.o` (the same one wired into `rms_gemv_rope`),
   so the core body becomes a single `func.call @rms_norm_2048_bf16`
   plus a `func.func private` declaration.  This adds +1 to
   `func.call` and adds the function decl to the segment, but the
   visible op counts of the rest of the IR are unchanged.

   Important: the PythoC `rms_norm_2048_bf16` kernel's positional
   signature is `(x, w, y, scratch)` -- the first arg is the input
   being squared in the accumulator, the second is the per-channel
   weight.  The cached AIR-stitched IR `func.call` argument order
   in the comparable `rms_gemv_rope` builder is `(weight, x, y,
   scratch)`, which silently swaps the two -- decode still passes
   the HF answer gate because RMSNorm weights initialize near 1.0
   (so `sqrt(mean(w^2)) ≈ 1`), but it is mathematically incorrect.
   This builder uses the correct `(x, w, y, scratch)` arg order;
   if `rms_gemv_rope.py` is ever reworked to match the cached AIR
   inline body exactly, mirror that fix there.

## How to regenerate

```bash
cd programming_examples/pythoc/llama32_1b
python3 -c '
import sys; sys.path.insert(0, ".")
from builders.o_gemv_ffn import build_o_gemv_ffn_module
with open("/tmp/ogf.mlir", "w") as f:
    f.write(build_o_gemv_ffn_module())
'
diff -u reference_mlir/o_gemv_ffn.npu.air.mlir /tmp/ogf.mlir \
    > tests/fixtures/o_gemv_ffn.diff
```

## How to enable the builder at runtime

```bash
PYTHOC_LLAMA_USE_PLACED_BUILDERS=o_gemv_ffn \
  PEANO_INSTALL_DIR=/path/to/llvm-aie \
  python3 llama32_1b_inference.py --compile-only

PYTHOC_LLAMA_USE_PLACED_BUILDERS=o_gemv_ffn \
  PEANO_INSTALL_DIR=/path/to/llvm-aie \
  python3 llama32_1b_inference.py --run-only --n-tokens 30 \
    --prompt "What is the capital of France?" \
    --hf-model-id unsloth/Llama-3.2-1B-Instruct
```

Enable the full decode placed-IRON cascade (all three builders):

```bash
PYTHOC_LLAMA_USE_PLACED_BUILDERS=lm_head_gemv,rms_gemv_rope,o_gemv_ffn \
  python3 llama32_1b_inference.py ...
```

The flag is read by `kernel_builder/aie_ir_gen.build_o_gemv_ffn_ir`.
Default is off so the cached path stays the steady state until all
Phase-4 kernels are ported.

## Key implementation gotchas

These are the bugs I burned through during the port; documenting so
the remaining Phase 4 builders (rms_gemms_rope, o_ffn) can skip them:

1. **15-arg runtime signature is shared across every segment** --
   each segment's runtime_sequence must accept exactly the same 15
   memrefs in the same order, even when most are unused by that
   segment.  The dispatcher's `aiex.run` symbol type-check enforces
   this.  See `builders/_emit.o_gemv_ffn_host_arg_types()`.

2. **K=8192 down-projection is its own variant.**  Same shape of
   matvec herd ([8,1]) and same mem-tile bandwidth, but the compute
   tile holds `M_TILE_K8192=2` output rows per call instead of 8,
   `K_TILE_K8192=1` (so no inner k loop step inside the core), and
   the L1 buffers are sized `2 × 8192` for weight, `8192` for input,
   `2` for output.  External func names are `dg_linalg_fill_bf16` /
   `dg_matvec_vectorized_bf16_bf16`, link_with = `mv_k8192_pythoc.o`.
   Output rows = 2048 across 8 outer iters (256 rows per outer).

3. **Eltwise-add segments have direct shim<->compute flows** --
   no mem tile.  Each compute tile owns 3 buffers of 256 bf16 (in1,
   in2, out) and a 6-lock barrier.  Input/output stride between
   columns is 256 (one tile per column).  Same lock/mem pattern as
   the rope-segments in rms_gemv_rope; only the body differs (inline
   `arith.addf` of 16-wide vector slices).

4. **SwiGLU uses a 1024-elt buffer per tile**, 8 tiles cover
   HIDDEN_DIM = 8192.  Input dimensions are `[(2, 512), (512, 1)]`
   (= 1024 contig per task).  Direct shim<->compute, link_with
   `silu_and_mul_bf16.o`.

5. **Core lock-action polarity is the complement of the mem DMA
   block.**  Bug pattern called out in rms_gemv_rope_diff.md still
   applies to every segment here; the 6-lock barrier (ids 5..0,
   init = 1,0,1,0,1,0) splits cleanly into "mem acquires id=2k+1,
   releases id=2k" and "core acquires id=2k, releases id=2k+1" for
   each of the 3 (input1, input2, output) buffer slots.

6. **Dispatcher firing order matches pipeline order** --
   og → a1 → rm → gg → ug → sw → dg → a2.  Reverse of the AIR emit
   order (which is reverse-topo).

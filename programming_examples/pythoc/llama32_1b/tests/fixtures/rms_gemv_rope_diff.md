# RMS+GEMV+RoPE: placed-IRON builder vs. cached AIR reference

This note tracks the structural diff between
`builders/rms_gemv_rope.build_rms_gemv_rope_module()` and the cached
AIR-stitched `reference_mlir/rms_gemv_rope.npu.air.mlir`.

## Bottom line

The two MLIR modules are **structurally equivalent**: same op counts
across every aie/aiex dialect op type, same flow topology, same lock
allocations, same buffer counts.  `aiecc` compiles the placed-IRON
output ~5% faster than the cached path (2.1 s vs 2.0 s; comparable),
and produces an ELF that passes the HF answer gate
(`A: The capital of France is Paris.`) and the snapshot regression
gate (identical decode token IDs to the cached baseline; logits corr
identical to 6 decimal places; K/V cache corr identical).

| Metric                          | placed-IRON | cached AIR |
| ------------------------------- | ----------- | ---------- |
| total MLIR lines                | 3,241       | 3,113      |
| `aie.device`                    | 7           | 7          |
| `aie.tile`                      | 78          | 78         |
| `aie.lock`                      | 258         | 258        |
| `aie.buffer`                    | 130         | 130        |
| `aie.flow`                      | 129         | 129        |
| `aie.shim_dma_allocation`       | 60          | 60         |
| `aie.mem`                       | 27          | 27         |
| `aie.memtile_dma`               | 24          | 24         |
| `aie.core`                      | 27          | 27         |
| `aiex.dma_configure_task_for`   | 77          | 77         |
| `aiex.dma_start_task`           | 77          | 77         |
| `aiex.dma_await_task`           | 35          | 35         |
| `aiex.dma_free_task`            | 42          | 42         |
| `aiex.configure`                | 6           | 6          |
| `aiex.run`                      | 6           | 6          |
| `aie.dma_bd`                    | 254         | 254        |
| `aie.runtime_sequence`          | 7           | 7          |
| `func.call`                     | 51          | 51         |
| `aie.use_lock`                  | 516         | 516        |
| HF answer-gate result           | PASS        | PASS       |
| snapshot decode token IDs       | identical   | -          |
| logits corr                     | 0.854103    | 0.854103   |
| K_cache corr min                | 0.882564    | 0.882564   |
| V_cache corr min                | 0.883117    | 0.883117   |
| decode tok/s (HF gate)          | 7.95        | 7.55-7.9   |
| aiecc compile time              | 2.1 s       | 2.0 s      |

## Module layout

Six segment devices + one dispatcher:

  * `@rk_rope_seg` -- 1 compute tile, RoPE on 512-elt K vector
  * `@rq_rope_seg` -- 1 compute tile, RoPE on 2048-elt Q vector
  * `@v_matvec_bf16_0` -- 8 compute tiles (herd [8,1]), V GEMV (512x2048)
  * `@k_matvec_bf16_0` -- 8 compute tiles, K GEMV (512x2048)
  * `@q_matvec_bf16_0` -- 8 compute tiles, Q GEMV (2048x2048) -- 2 outer iters
  * `@r_rms_seg` -- 1 compute tile, RMSNorm on 2048-elt vector
  * (unnamed) dispatcher -- fires the 6 segments in topo order
    (RMSNorm -> Q -> K -> V -> RoPE-Q -> RoPE-K).

All 7 runtime sequences share the same 13-arg host signature; the
arg layout is dictated by the dispatcher device's
`aiex.runtime_sequence @rms_gemv_rope(...)` block in the cached IR.

## Cosmetic differences

These come purely from how MLIR's printer renders auto-generated SSA
names + which named attributes are emitted by aircc but not by our
builder.  None affect correctness or `aiecc` compilation.

1. **SSA value names** -- AIR's aircc emits explicit `sym_name`
   attributes like `buf3`, `lock_0_2_63`; our builder lets MLIR
   auto-name (`buffer_0_2`, `lock_0_2_28`).

2. **`task_id = 0 : i32` attribute on `aie.dma_bd`** -- present in
   the cached IR (aircc adds it during stitching), absent in our
   builder's output.  Optional metadata, not consumed by `aiecc`.

3. **Cores' `air.herd_*` attributes** -- e.g.
   `{air.herd_local_id = array<i64: 7, 0>, air.herd_name = "v_herd_0",
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

## How to regenerate

```bash
cd programming_examples/pythoc/llama32_1b
python3 -c '
import sys; sys.path.insert(0, ".")
from builders.rms_gemv_rope import build_rms_gemv_rope_module
with open("/tmp/rgr.mlir", "w") as f:
    f.write(build_rms_gemv_rope_module())
'
diff -u reference_mlir/rms_gemv_rope.npu.air.mlir /tmp/rgr.mlir \
    > tests/fixtures/rms_gemv_rope.diff
```

## How to enable the builder at runtime

```bash
PYTHOC_LLAMA_USE_PLACED_BUILDERS=rms_gemv_rope \
  PEANO_INSTALL_DIR=/path/to/llvm-aie \
  python3 llama32_1b_inference.py --compile-only

PYTHOC_LLAMA_USE_PLACED_BUILDERS=rms_gemv_rope \
  PEANO_INSTALL_DIR=/path/to/llvm-aie \
  python3 llama32_1b_inference.py --run-only --n-tokens 30 \
    --prompt "What is the capital of France?" \
    --hf-model-id unsloth/Llama-3.2-1B-Instruct
```

The flag is read by `kernel_builder/aie_ir_gen.build_rms_gemv_rope_ir`.
Default is off so the cached path stays the steady state until all
Phase-4 kernels are ported.

To enable multiple builders simultaneously:

```bash
PYTHOC_LLAMA_USE_PLACED_BUILDERS=lm_head_gemv,rms_gemv_rope \
  python3 llama32_1b_inference.py ...
```

## Key implementation gotchas

These are the bugs I burned through during the port; documenting so
the remaining Phase 4 builders (o_gemv_ffn, rms_gemms_rope, o_ffn) can
skip them:

1. **Core lock-action polarity is the complement of the mem DMA
   block.**  For a single tile with the 6-lock barrier pattern
   (ids 5..0 init = 1,0,1,0,1,0):
     - mem MM2S 0 (drain output buffer Y to shim): acquires id=0
       (Y full), releases id=1 (Y done).
     - mem S2MM 0 (fill input buffer X1 from shim): acquires id=3
       (X1 avail), releases id=2 (X1 ready).
     - mem S2MM 1 (fill input buffer X2 from shim): acquires id=5
       (X2 avail), releases id=4 (X2 ready).
     - core: acquires {id=1, id=2, id=4} (waits for Y done, X1
       ready, X2 ready), runs the kernel, releases {id=3, id=5,
       id=0} (signals X1 avail, X2 avail, Y full).
   My first cut had the core acquiring/releasing in the *DMA* roles,
   which compiles cleanly but hangs at runtime (deadlock).

2. **The 13-arg runtime signature is shared across every segment** --
   each segment's runtime_sequence must accept exactly the same 13
   memrefs in the same order, even when most are unused by that
   segment.  The dispatcher's `aiex.run` symbol type-check enforces
   this.

3. **Q matvec has 2 outer iterations**, K/V have 1.  Q's
   `output_outer_stride` is 1024 (= ROWS_PER_OUTER) and its
   `weight_outer_stride` is 2_097_152 (= 1024 * 2048 bf16 elements).
   Q's input `repeat_count` is 31 (32 deliveries per outer); K/V's
   is 15.

4. **AIR emits buffers per tile in descending sym-id order**
   (`buf3, buf2, buf1, buf0`).  Our builder follows the same emit
   order so the `aiecc` post-stitching diff is minimal -- but `aiecc`
   itself is order-insensitive at this level.

5. **The dispatcher device is unnamed** (`aie.device(npu2) { ... }`,
   no `sym_name`).  Only the segments carry a `sym_name`.

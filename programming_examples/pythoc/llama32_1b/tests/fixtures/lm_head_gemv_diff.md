# LM Head GEMV: placed-IRON builder vs. cached AIR reference

This note tracks the structural diff between
`builders/lm_head_gemv.build_lm_head_gemv_module()` and the cached
AIR-stitched `reference_mlir/lm_head_gemv.npu.air.mlir`.

## Bottom line

The two MLIR modules are **structurally equivalent**: same ops, same
attributes (modulo a small set of cosmetic differences listed below),
same DMA task counts (6528 in both), same flow topology, same lock
allocations. `aiecc` compiles both into an ELF that passes the HF
answer gate ("A: The capital of France is Paris.") and produces
identical decode token IDs vs the cached path.

| Metric                          | placed-IRON | cached AIR |
| ------------------------------- | ----------- | ---------- |
| total MLIR lines                | 19,886      | 19,567     |
| `aiex.dma_*` total occurrences  | 6,528       | 6,528      |
| `aie.flow` occurrences          | 256         | 256        |
| `aie.shim_dma_allocation`       | 136         | 136        |
| `aie.lock`                      | 512         | 512        |
| `aie.buffer`                    | 320         | 320        |
| HF answer-gate result            | PASS        | PASS       |
| snapshot decode token IDs        | identical   | -          |

## Cosmetic differences

These come purely from how MLIR's printer renders auto-generated SSA
names + which named attributes are emitted by aircc but not by our
builder.  None of them affect correctness or `aiecc` compilation.

1. **SSA value names** -- AIR's aircc emits explicit `sym_name`
   attributes like `buf303`, `lock_7_2_63`; our builder lets MLIR
   auto-name (`buffer_0_2`, `lock_0_2_28`).  No semantic impact.

2. **Per-partition core / mem ordering** -- aircc emits cores in
   descending column order (`mem_7_2`, then `mem_6_2`, ...,
   `mem_0_2`) because the AIR herd unrolling is bottom-to-top.  Our
   builder emits them in ascending order (`mem_0_2`, ..., `mem_7_2`).
   `aiecc` is order-insensitive at this level.

3. **`task_id = 0 : i32` attribute on `aie.dma_bd`** -- present in
   the cached IR (aircc adds it during stitching), absent in our
   builder's output.  Optional metadata, not consumed by `aiecc`.

4. **Cores' `air.herd_*` attributes** -- e.g.
   `{air.herd_local_id = array<i64: 7, 0>, air.herd_name = "p7_herd_0",
   air.herd_size = array<i64: 8, 1>}`.  AIR-only metadata; not needed
   downstream of stitching.

5. **`#loop_annotation = #llvm.loop_annotation<mustProgress = true>`**
   -- a top-level attr aircc attaches via its `LowerHerds` pass.
   Optional; aiecc accepts the body whether or not it is annotated.

6. **dispatcher device's runtime_sequence basic-block ordering** --
   the cached IR's mem DMA basic blocks are emitted in a slightly
   different topological order than ours; the CFG itself is identical
   (same predecessors / successors).

## How to regenerate

```bash
cd programming_examples/pythoc/llama32_1b
python3 -c '
import sys; sys.path.insert(0, ".")
from builders.lm_head_gemv import build_lm_head_gemv_module
with open("/tmp/lm.mlir", "w") as f:
    f.write(build_lm_head_gemv_module())
'
diff -u reference_mlir/lm_head_gemv.npu.air.mlir /tmp/lm.mlir > tests/fixtures/lm_head_gemv.diff
```

## How to enable the builder at runtime

```bash
PYTHOC_LLAMA_USE_PLACED_BUILDERS=lm_head_gemv \
  PEANO_INSTALL_DIR=/path/to/llvm-aie \
  python3 llama32_1b_inference.py --compile-only
```

The flag is read by `kernel_builder/aie_ir_gen.build_lm_head_gemv_ir`.
Default is off so the cached path stays the steady state until all
phase-4 kernels are ported.

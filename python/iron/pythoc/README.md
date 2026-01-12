# PythoC kernels in IRON

This package lets you bring PythoC kernels into IRON programs either as pre-compiled sources or inline, single-source kernels.

## Prerequisites
- PythoC importable (`pip install pythoc` or set `PYTHOC_PATH` to the checkout).
- AIE-aware `llc` available; set `LLVM_AIE_BIN` if it is not in `/scratch/jefff/acdc/aie-venv/lib/python3.12/site-packages/llvm-aie/bin`.
- Peano toolchain installed for downstream IRON compilation.

## External kernels (separate .py files)
1. Author a kernel in PythoC syntax (see `PythoC/pythoc_kernels/` for examples).
2. Compile it to an object file:
   ```python
   from aie.iron.pythoc import compile_pythoc_kernel, PythocKernel
   obj = compile_pythoc_kernel("pythoc_kernels/mul.py", "eltwise_mul_vectorized_i32", target_arch="aie2")
   kernel = PythocKernel("eltwise_mul_vectorized_i32", str(obj), [tile_ty, tile_ty, tile_ty, np.int32])
   ```
   `compile_pythoc_kernel` also emits a `.ll` alongside the `.o`.
3. Pass the `PythocKernel` into your `Worker`; its `bin_name` is picked up via `link_with` when the core is built.

## Inline kernels (single source)
1. Define the kernel inline and mark it with `@aie_kernel`:
   ```python
   from aie.iron.pythoc import aie_kernel, PythocKernel
   from pythoc import ptr, i32
   from pythoc.aie.operations import load_v, store_v, vector_add
   from pythoc.aie.vector import aie_vector

   @aie_kernel
   def add_kernel(a: ptr[i32, True], b: ptr[i32, True], c: ptr[i32, True], N: i32):
       vec = 16
       for i in range(N // vec):
           off = i * vec
           store_v(c + off, vector_add(load_v(a + off, vec), load_v(b + off, vec)))

   kernel = PythocKernel(add_kernel, [tile_ty, tile_ty, tile_ty, np.int32], target_arch="aie2p")
   ```
   The decorator captures the source, and `PythocKernel` compiles it to an object file in a temporary directory.

## Type utilities
- `pythoc_to_numpy_type(type_str)` maps PythoC type strings to `numpy.dtype`.
- `infer_kernel_signature(fn)` converts PythoC-annotated function parameters to the IRON type list expected by `Kernel`.

## Examples
- External kernel flow: `programming_examples/pythoc/vector_mul_pythoc.py`
- Inline single-file flow: `programming_examples/pythoc/vector_add_inline.py`

# MDV6 → PythoC conversion pattern

How to port `programming_examples/ml/mdv6/<layer>/` (IRON + external C++ kernel)
into a single-file PythoC + IRON example matching the `pythoc/` style.

The first worked example is `mdv6/prototypes/elementwise/elementwise_pythoc.py` — start
there before applying this to another layer.

---

## Two viable approaches

### Pattern A — Reuse the existing `.cc` kernel via `PythocKernel(name, .o)`

`PythocKernel` accepts the same arguments as `Kernel`, so the only change
needed from the original `aie2.py` is a one-line swap. The C++ kernel still
has to be compiled to a `.o` (with `clang` + `PEANOWRAP2P_FLAGS`) outside
the script (or via a small helper that shells out to `clang`).

Use this when:
- The layer kernel is large or uses C++-only features (templates, SIMD
  intrinsics not yet exposed in PythoC, packed vector accessors).
- You want the example to demonstrate the IRON ObjectFifo / Worker layout
  rather than the kernel itself.

### Pattern B — Re-implement the kernel inline with `@aie_kernel`

Replace the `Kernel("foo", "foo.o", [...])` reference with an inline
`@aie_kernel`-decorated PythoC function, then build a `PythocKernel` from it.
This is what `mdv6/prototypes/elementwise/elementwise_pythoc.py` does for add/mul/max.

Use this when:
- The kernel is small (≲100 lines) or naturally maps to a handful of
  `vector_add` / `vector_mul` / `vmax_ltbf16` / etc. primitives.
- The PythoC intrinsics you need are exposed (see "Intrinsic gotchas" below).

Both patterns produce a runnable single-file example. Pick A first if the
kernel is large or uses anything exotic.

---

## Canonical single-file structure (Pattern B)

Skeleton — see `mdv6/prototypes/elementwise/elementwise_pythoc.py` for the full version.

```python
#!/usr/bin/env python3
# REQUIRES: ryzen_ai_npu2
# RUN: %python %s --device npu2 ... --work-dir ./<layer>_pythoc_build | FileCheck %s
# CHECK: PASS!

import argparse, sys
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import ObjectFifo, Program, Runtime, Worker
import aie.iron as iron
from aie.utils.compile import compile_mlir_module
from aie.iron.device import NPU2Col1
from aie.iron.pythoc import aie_kernel, PythocKernel
from aie.utils import DefaultNPURuntime, NPUKernel

from pythoc import ptr, i32, bf16
from pythoc.aie import aie_vector, load_v, store_v, vector_add  # + others as needed
from pythoc.aie.profiling import event0, event1


@aie_kernel
def my_kernel(a: ptr[bf16, True], c: ptr[bf16, True], n: i32):
    event0()
    # ... PythoC body ...
    event1()


def build_mlir_module(device, size: int):
    tensor_ty = np.ndarray[(size,), np.dtype[np.uint16]]  # bf16 carried as uint16

    kernel = PythocKernel(my_kernel, [tensor_ty, tensor_ty, np.int32])
    of_in  = ObjectFifo(tensor_ty, depth=1, name="in")
    of_out = ObjectFifo(tensor_ty, depth=1, name="out")

    def core_fn(of_in, of_out, kernel):
        ein  = of_in.acquire(1)
        eout = of_out.acquire(1)
        kernel(ein, eout, size)
        of_in.release(1)
        of_out.release(1)

    worker  = Worker(core_fn, [of_in.cons(), of_out.prod(), kernel])
    runtime = Runtime()
    with runtime.sequence(tensor_ty, tensor_ty) as (A, C):
        runtime.start(worker)
        runtime.fill(of_in.prod(), A)
        runtime.drain(of_out.cons(), C, wait=True)

    module = Program(device, runtime).resolve_program()
    assert module.operation.verify()
    return module


def run_with_xrt(xclbin, insts, size):
    handle = DefaultNPURuntime.load(NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE"))
    # build bf16 input as uint16, run, compare against numpy reference
    ...


def main():
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    module = build_mlir_module(NPU2Col1(), args.size)
    with open(args.work_dir / "kernel.mlir", "w") as fh: print(module, file=fh)
    compile_mlir_module(mlir_module=module,
                       insts_path=str(args.work_dir / "insts.bin"),
                       xclbin_path=str(args.work_dir / "final.xclbin"),
                       work_dir=str(args.work_dir), verbose=args.verbose)
    actual, expected = run_with_xrt(args.work_dir / "final.xclbin",
                                    args.work_dir / "insts.bin", args.size)
    if np.allclose(actual, expected, rtol=1e-2, atol=1e-2):
        print("PASS!"); return 0
    print("FAILED"); return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## Intrinsic gotchas (PythoC bf16)

These are what bit me on `elementwise` and the subsequent layer ports:

1. **`vector_max` only supports signed integers.** It generates
   `icmp_signed('>', a, b); select(...)`, which is invalid for `bf16`.
   Use `vmax_ltbf16(a, b)` instead — it returns `(max_vec, lt_mask)`,
   so unpack as `vc, _mask = vmax_ltbf16(va, vb)`. See `PythoC/pythoc_kernels/relu.py`.

2. **`@aie_kernel` ships a fixed global set.** The compiler in
   `install/mlir-aie/python/aie/iron/pythoc/compiler.py` only auto-imports
   `vector_add`/`vector_mul`/`vector_sub`/`vector_max`/`vector_min` and a
   handful of others — `vmax_ltbf16`, `invsqrt`, `getTanhBf16`, etc. are
   NOT in it. To use them, pass `extra_globals={"vmax_ltbf16": vmax_ltbf16}`
   (etc.) when constructing `PythocKernel(...)`. See `elementwise_pythoc.py`.

3. **bf16 vector width.** Use 32-element vectors (`aie_vector[bf16, 32]`)
   for bf16 on AIE2P — that matches the C++ `v32bfloat16` and the
   `vmax_ltbf16` signature. (16-wide `bf16` vectors work for add/mul too —
   and are sometimes preferable: see gotcha #6.)

4. **bf16 ↔ uint16 plumbing.** IRON ObjectFifos and `iron.tensor` use
   numpy `uint16`. Convert with `bf16_arr.view(np.uint16)` going in and
   `np.array(out_u16, np.uint16).view(bfloat16).astype(np.float32)` coming
   out, then compare in fp32.

5. **Float-precision reference.** Always cast both sides to `bf16` then
   `float32` before `np.allclose` (rtol/atol ≈ 1e-2) — pure-fp32 reference
   will spuriously fail on values that bf16 rounds. For chains with
   `invsqrt`/`fast_sigmoid` approximations, relax to 5e-2 or higher.

6. **LLVM auto-vectorizer can produce illegal AIE2P codegen.** Three
   independent ports (`bottleneck`, `repconv`, `repncsp`) hit this:
   a scalar bf16 loop like
   ```python
   while i < n:
       c[i] = bf16(f32(a[i]) + f32(b[i]))
       i = i + 1
   ```
   gets auto-vectorized by `opt -O2` into `fadd <32 x bfloat>` (or worse,
   a `<32 x float> → <32 x bf16>` `fptrunc`), which the AIE2P llc/GISel
   legalizer rejects (`unable to legalize G_FADD <32 x s16>` /
   `G_FPTRUNC not legal`). The C++ kernels escape this because Clang
   lowers bf16 differently; llvmlite produces native bf16 fadd that opt
   happily packs.

   **Fix (preferred):** Rewrite the loop with explicit 16-wide `vector_add` /
   `vector_mul` calls. See `bottleneck_pythoc.py` and `repconv_pythoc.py`.

   **Fix (escape hatch, fragile):** Disable the vectorizer by patching
   `subprocess.run` to inject `-vectorize-loops=false -vectorize-slp=false`
   into opt calls. See top of `repncsp_pythoc.py`. This is process-global
   and affects every subprocess the script invokes — use only if explicit
   vectorization is impractical. The upstream fix would be a
   `PYTHOC_OPT_FLAGS` env hook in
   `install/mlir-aie/python/aie/iron/pythoc/compiler.py`.

7. **PythoC scope rules.** A name can't be re-declared in the same scope —
   if you have repeated loop variables across stages, suffix them
   (`i1`, `i2`, `i3`). See `sppelan_pythoc.py`.

8. **Calling vector intrinsics on scalars.** If you only need a scalar
   result (e.g. when channel count < vector width), broadcast the scalar
   into a vector, call the intrinsic, then `extract_elem(vec, 0)`. See
   `sppelan_pythoc.py` for the SiLU-on-scalar pattern.

9. **Numerical approximations.** The original C++ kernels use cheap
   approximations:
   - `fast_sqrt` (Quake bit-hack Newton-Raphson) for BN normalization
   - `fast_sigmoid(x) = 0.5 + x/(2*(1+|x|))` for SiLU

   Two valid porting strategies:
   - **Mirror the C++ exactly** (use `bitcast_i32_to_f32` for the sqrt
     bit-hack, replicate fast_sigmoid). The numpy reference must match,
     so write it with the same approximation. See `aconv_pythoc.py`.
   - **Use AIE intrinsics** (`invsqrt`, `getTanhBf16`-based sigmoid).
     More accurate but the numpy reference must also use the matching
     formula (or tolerance must be relaxed). See `batchnorm_silu_pythoc.py`.

10. **BN host-folding.** `gamma*(x-mean)/sqrt(var+eps) + beta` collapses
    to `w'*x + b'`. Pre-fold on the host so the device kernel never needs
    `sqrt` or division. See `elan_pythoc.py`, `sppelan_pythoc.py`,
    `repncsp_elan_pythoc.py`.

11. **Tile program memory is 16 KB.** Big composite layers
    (`repncsp_elan`) may not fit if each variant is a separate helper
    kernel — merge variants with a flag parameter (`do_silu: i32`)
    instead of duplicating code.

---

## Step-by-step recipe (per layer)

1. **Inspect the original.** Read `aie2.py` (data layout, ObjectFifo
   shapes, kernel signature) and `<layer>_bf16.cc` (what the kernel
   actually computes).

2. **Pick the pattern.** If `<layer>_bf16.cc` is small + uses only
   add/mul/max/relu-style ops, try Pattern B. If it's a multi-hundred-line
   layer fusion (bottleneck, sppelan, repncsp_elan, conv variants, gemm),
   start with Pattern A (`PythocKernel(name, ".o")`) to get the IRON wiring
   right; port the kernel body later if desired.

3. **Write the single file** at
   `pythoc/mdv6/<layer>/<layer>_pythoc.py`. Drop the IRON `Kernel(...)`
   call in favor of `PythocKernel`.

4. **Test data + reference.** Use `np.random.default_rng(42)`,
   convert via `ml_dtypes.bfloat16`, compute a numpy reference, compare.

5. **Run on hardware.**
   ```bash
   source /home/jfifield/npu-dev-pythoc/env.sh
   cd mlir-aie/programming_examples/pythoc/mdv6/<layer>
   python <layer>_pythoc.py --device npu2 [...]
   ```
   The script must print `PASS!` on stdout for the FileCheck line.

6. **Delete the original `aie2.py`, `Makefile`, `*.cc`, `run.lit`, `test.py`** —
   the single PythoC file replaces them all. (Or keep the `.cc` if you
   used Pattern A; in that case the script needs to ensure it's compiled
   before constructing the kernel.)

---

## Build / iteration tips

- Build artifacts (`<layer>_pythoc_build/`) are gitignored siblings of
  the script; safe to nuke between runs.
- First compile of a new kernel takes ~30–60 s (aiecc + peano).
  Subsequent runs with the same MLIR are faster.
- If you get `NameError: Variable 'foo' not defined` from PythoC's AST
  visitor, you're hitting gotcha #2 — add the symbol to `extra_globals`.
- If shapes don't match, double-check `ObjectFifo(tensor_ty, ...)` matches
  the kernel's `ptr[bf16]` access pattern and that `runtime.sequence(...)`
  arg types match `runtime.fill/drain` calls.

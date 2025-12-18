# PythoC IRON Integration Examples

This directory contains examples demonstrating how to use PythoC-compiled kernels in IRON programs.

## Overview

PythoC is a Python-to-LLVM compiler that generates AIE-compatible code. The IRON integration allows PythoC kernels to be used seamlessly in IRON programs alongside C++ kernels.

## Phase 1: External Kernels

Phase 1 demonstrates using pre-existing PythoC kernels compiled to LLVM IR.

### Example: `vector_mul_pythoc.py`

Demonstrates the complete workflow:
1. Compile a PythoC kernel (`mul.py`) to LLVM IR
2. Create an IRON `PythocKernel` object
3. Use the kernel in an IRON program with ObjectFIFOs
4. Generate MLIR for the design

**Run the example:**
```bash
# Source AIE environment first
source /scratch/jefff/acdc/env.sh

cd mlir-aie/programming_examples/pythoc
python3 vector_mul_pythoc.py
```

**Expected output:**
- Compilation of PythoC kernel to LLVM IR
- Creation of IRON Kernel object
- Generated MLIR design (first 50 lines)

**Note**: This example requires the full AIE/IRON environment. For a simpler test of just the compilation functionality, see `mlir-aie/python/iron/pythoc/test_standalone.py` which works without environment setup.

## Phase 2: Inline Kernels (Coming Soon)

Phase 2 will enable single-source programs where PythoC kernels are defined inline using the `@aie_kernel` decorator.

## Available PythoC Kernels

Located in `/scratch/jefff/acdc/PythoC/pythoc_kernels/`:

**Arithmetic** (no event markers - recommended for testing):
- `mul.py` - Element-wise multiplication
- `scale.py` - Vector scaling
- `bitwiseAND.py`, `bitwiseOR.py` - Bitwise operations
- `threshold.py` - Image thresholding

**With event markers** (may have compatibility issues):
- `add.py` - Element-wise addition
- `relu.py` - ReLU activation
- `reduce_add.py`, `reduce_max.py`, `reduce_min.py` - Reductions

**Image processing**:
- `rgba2gray.py`, `gray2rgba.py` - Color space conversion
- `addWeighted.py` - Weighted blending

**BFloat16 activations**:
- `silu.py`, `gelu.py`, `swiglu.py` - Activation functions
- `bf16_exp.py`, `bf16_softmax.py` - Math functions

## Integration Pattern

```python
from aie.iron.pythoc import compile_pythoc_kernel, PythocKernel
from aie.dialects.aie import *
import numpy as np

# 1. Compile PythoC kernel
ll_file = compile_pythoc_kernel(
    "path/to/kernel.py",
    "function_name",
    target_arch="aie2"
)

# 2. Create IRON kernel
tile_ty = np.ndarray[(256,), np.dtype[np.int32]]
kernel = PythocKernel(
    "function_name",
    str(ll_file),
    [tile_ty, tile_ty, tile_ty]  # Type signature
)

# 3. Use in IRON program
@core(ComputeTile, link_with=str(ll_file))
def core_body():
    elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
    function_name(elem_in, elem_out, size)  # Call kernel
    of_in.release(ObjectFifoPort.Consume, 1)
    of_out.release(ObjectFifoPort.Produce, 1)
```

## Type Mapping

PythoC types map to NumPy types for IRON:

| PythoC Type | NumPy Type | IRON Usage |
|-------------|------------|------------|
| `i32` | `np.int32` | Scalar parameter |
| `bf16` | `np.bfloat16` | Scalar parameter |
| `ptr[i32, True]` | `np.ndarray[(...,), np.dtype[np.int32]]` | Array parameter |
| `aie_vector[i32, 16]` | `np.ndarray[(16,), np.dtype[np.int32]]` | Vector parameter |

## Notes

- PythoC kernels compile to LLVM IR (`.ll` files)
- IRON's Peano toolchain compiles `.ll` → `.o` during build
- Event markers (`event0()`, `event1()`) may have target compatibility issues
- Use kernels without event markers for initial testing

## References

- PythoC Documentation: `/scratch/jefff/acdc/PythoC/README.md`
- IRON Programming Guide: `mlir-aie/guide/02_iron_programming_model.md`
- Integration Plan: `/scratch/jefff/acdc/PythoC/PYTHOC_IRON_INTEGRATION_PLAN.md`

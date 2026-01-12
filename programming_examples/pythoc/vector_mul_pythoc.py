#!/usr/bin/env python3
# vector_mul_pythoc.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

"""Example: Using PythoC kernels in IRON programs (Phase 1 - External kernels).

This example demonstrates how to:
1. Compile a PythoC kernel to LLVM IR
2. Create an IRON Kernel from the compiled IR
3. Verify the generated LLVM IR


Note: This is a demonstration of the integration pattern.
Full end-to-end execution requires additional IRON runtime setup.
"""

from pathlib import Path
import sys
import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

# Import PythoC IRON integration
from aie.iron.pythoc import compile_pythoc_kernel, PythocKernel


def main():
    # Configuration
    num_elements = 1024
    tile_size = 256
    vec_factor = 16
    
    # Type definitions
    tensor_ty = np.ndarray[(num_elements,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
    
    # Step 1: Compile PythoC kernel to LLVM IR
    print("Step 1: Compiling PythoC kernel...")
    pythoc_kernel_path = Path("/scratch/jefff/acdc/PythoC/pythoc_kernels/mul.py")
    
    ll_file = compile_pythoc_kernel(
        str(pythoc_kernel_path),
        function_name="eltwise_mul_vectorized_i32",
        target_arch="aie2",
        verbose=True
    )
    
    print(f"✓ LLVM IR generated: {ll_file}\n")
    
    # Step 2: Create IRON Kernel from compiled IR
    print("Step 2: Creating IRON Kernel...")
    mul_kernel = PythocKernel(
        "eltwise_mul_vectorized_i32",
        str(ll_file),
        [tile_ty, tile_ty, tile_ty, np.int32]  # Types: input_a, input_b, output, size
    )
    
    print(f"✓ PythocKernel created: {mul_kernel._name}")
    print(f"✓ Kernel bin_name: {mul_kernel.bin_name}")
    print(f"✓ Target arch: {target_arch}\n")
    
    # Step 3: Generate MLIR design using the kernel
    print("Step 3: Generating MLIR design...")
    
    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu1_1col)
        def device_body():
            # Tile declarations
            ShimTile = tile(0, 0)
            ComputeTile = tile(0, 2)
            
            # ObjectFIFOs for data movement
            of_in_a = object_fifo("in_a", ShimTile, ComputeTile, 2, tile_ty)
            of_in_b = object_fifo("in_b", ShimTile, ComputeTile, 2, tile_ty)
            of_out = object_fifo("out", ComputeTile, ShimTile, 2, tile_ty)
            
            # Resolve kernel to create MLIR function declaration
            mul_kernel.resolve()
            
            # Core program using PythoC kernel
            @core(ComputeTile, link_with=str(ll_file))
            def core_body():
                for _ in range_(sys.maxsize):
                    # Process 4 tiles (1024 elements / 256 per tile)
                    for _ in range_(4):
                        elem_a = of_in_a.acquire(ObjectFifoPort.Consume, 1)
                        elem_b = of_in_b.acquire(ObjectFifoPort.Consume, 1)
                        elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                        
                        # Call PythoC kernel
                        # Note: In actual MLIR, this becomes a function call
                        # The kernel signature is: (ptr, ptr, ptr, i32)
                        mul_kernel(elem_a, elem_b, elem_out, tile_size)
                        
                        of_in_a.release(ObjectFifoPort.Consume, 1)
                        of_in_b.release(ObjectFifoPort.Consume, 1)
                        of_out.release(ObjectFifoPort.Produce, 1)
            
            # Runtime sequence
            @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(
                    metadata=of_in_a,
                    bd_id=0,
                    mem=A,
                    sizes=[1, 1, 1, num_elements]
                )
                npu_dma_memcpy_nd(
                    metadata=of_in_b,
                    bd_id=1,
                    mem=B,
                    sizes=[1, 1, 1, num_elements]
                )
                npu_dma_memcpy_nd(
                    metadata=of_out,
                    bd_id=2,
                    mem=C,
                    sizes=[1, 1, 1, num_elements]
                )
                dma_wait(of_in_a, of_in_b, of_out)
        
        # Print generated MLIR
        print("✓ MLIR design generated\n")
        print("=" * 60)
        print("Generated MLIR (excerpt):")
        print("=" * 60)
        mlir_str = str(ctx.module)
        # Print first 50 lines
        lines = mlir_str.split('\n')[:50]
        print('\n'.join(lines))
        if len(mlir_str.split('\n')) > 50:
            print(f"\n... ({len(mlir_str.split('\n')) - 50} more lines)")
        print("=" * 60)
    
    print("\n✓ SUCCESS: PythoC kernel integrated into IRON program")
    print("\nNext steps:")
    print("- Compile MLIR to xclbin using aiecc")
    print("- Run on NPU hardware using pyxrt")
    print("- See mlir-aie/programming_guide for full examples")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

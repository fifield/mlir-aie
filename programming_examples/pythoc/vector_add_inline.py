# vector_add_inline.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

"""
IRON + PythoC Phase 2 Example: Inline Kernel Definition

This example demonstrates single-source development with PythoC kernels
defined inline using the @aie_kernel decorator. The kernel is compiled
automatically when creating the PythocKernel instance.
"""

import numpy as np
import sys

from aie.iron import Program, Worker, ObjectFifo, Runtime
from aie.iron.device import NPU1Col1
from aie.iron.placers import SequentialPlacer
from aie.iron.pythoc import aie_kernel, PythocKernel

# Import PythoC types and operations for kernel definition
from pythoc import ptr, i32
from pythoc.aie.operations import load_v, store_v, vector_add
from pythoc.aie.vector import aie_vector


# Define kernel inline with @aie_kernel decorator
@aie_kernel
def add_kernel(a: ptr[i32, True], b: ptr[i32, True], 
               c: ptr[i32, True], N: i32):
    """Vectorized element-wise addition kernel.
    
    Adds two int32 arrays element-wise using AIE2 vector operations.
    Processes 16 elements per iteration using 512-bit vectors.
    
    Args:
        a: Input array A (restrict pointer)
        b: Input array B (restrict pointer)
        c: Output array C (restrict pointer)
        N: Number of elements to process
    """
    vec_size: i32 = 16  # AIE2 processes 16 x i32 per vector
    iterations: i32 = N // vec_size
    
    # Use pointer arithmetic (PythoC pattern)
    pA: ptr[i32] = a
    pB: ptr[i32] = b
    pC: ptr[i32] = c
    
    i: i32 = 0
    while i < iterations:
        # Load 16 elements from each input
        va: aie_vector[i32, 16] = load_v(pA, 16)
        vb: aie_vector[i32, 16] = load_v(pB, 16)
        
        # Perform vectorized addition
        vc: aie_vector[i32, 16] = vector_add(va, vb)
        
        # Store result
        store_v(pC, vc)
        
        # Advance pointers
        pA = pA + vec_size
        pB = pB + vec_size
        pC = pC + vec_size
        i = i + 1


def main():
    """Build and run IRON program with inline PythoC kernel."""
    
    # Configuration
    N = 256  # Must be multiple of 16 for vectorization
    tile_size = N
    
    # Define tile type for ObjectFifos
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
    
    # Create PythocKernel from decorated function
    # The kernel is compiled automatically during initialization
    kernel = PythocKernel(
        add_kernel,  # Pass decorated function directly
        [tile_ty, tile_ty, tile_ty, np.int32]  # Type signature
    )
    
    # Define ObjectFifos for data movement
    of_a = ObjectFifo(tile_ty, name="in_a")
    of_b = ObjectFifo(tile_ty, name="in_b")
    of_c = ObjectFifo(tile_ty, name="out")
    
    # Define core function that uses the kernel
    def core_fn(of_a, of_b, of_c, kernel):
        """AIE core function: acquire buffers, call kernel, release."""
        elem_a = of_a.acquire(1)
        elem_b = of_b.acquire(1)
        elem_c = of_c.acquire(1)
        
        # Call PythoC kernel
        kernel(elem_a, elem_b, elem_c, tile_size)
        
        of_a.release(1)
        of_b.release(1)
        of_c.release(1)
    
    # Create worker with kernel
    worker = Worker(
        core_fn,
        [of_a.cons(), of_b.cons(), of_c.prod(), kernel]
    )
    
    # Build IRON program
    with Program(NPU1Col1(), SequentialPlacer()):
        runtime = Runtime()
        
        # Connect ObjectFifos to runtime
        runtime.buffer(of_a).connect(worker)
        runtime.buffer(of_b).connect(worker)
        worker.connect(runtime.buffer(of_c))
    
    # Prepare test data
    a_data = np.arange(N, dtype=np.int32)
    b_data = np.arange(N, dtype=np.int32) * 2
    c_data = np.zeros(N, dtype=np.int32)
    
    # Execute on NPU
    print(f"Running inline PythoC kernel: {add_kernel.__name__}")
    print(f"Processing {N} elements...")
    
    runtime(a_data, b_data, c_data)
    
    # Verify results
    expected = a_data + b_data
    if np.allclose(c_data, expected):
        print("✓ Test PASSED")
        print(f"  Sample: {a_data[:4]} + {b_data[:4]} = {c_data[:4]}")
        return 0
    else:
        print("✗ Test FAILED")
        print(f"  Expected: {expected[:8]}")
        print(f"  Got:      {c_data[:8]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

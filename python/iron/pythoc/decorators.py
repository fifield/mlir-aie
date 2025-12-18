# decorators.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

"""Decorators for inline PythoC kernel definitions.

This module provides the @aie_kernel decorator for defining PythoC kernels
inline within IRON programs, enabling single-source development.
"""

import inspect
from typing import Callable


def aie_kernel(fn: Callable) -> Callable:
    """Decorator to mark Python function for PythoC AIE compilation.
    
    This decorator marks a function for compilation with PythoC to AIE2
    machine code. The decorated function can be used directly with
    PythocKernel to create an IRON kernel without needing a separate
    kernel file.
    
    The function must use PythoC type annotations and AIE intrinsics.
    
    Usage:
        from aie.iron.pythoc import aie_kernel, PythocKernel
        from pythoc import ptr, i32
        from pythoc.aie.operations import load_v, store_v, vector_add
        from pythoc.aie.vector import aie_vector
        
        @aie_kernel
        def add_kernel(a: ptr[i32, True], b: ptr[i32, True], 
                       c: ptr[i32, True], N: i32):
            vec_size: i32 = 16
            iterations: i32 = N // vec_size
            for i in range(iterations):
                offset: i32 = i * vec_size
                va: aie_vector[i32, 16] = load_v(a + offset, 16)
                vb: aie_vector[i32, 16] = load_v(b + offset, 16)
                vc: aie_vector[i32, 16] = vector_add(va, vb)
                store_v(c + offset, vc)
        
        # Use in IRON
        kernel = PythocKernel(add_kernel, [tile_ty, tile_ty, tile_ty, np.int32])
    
    Args:
        fn: Function to decorate
        
    Returns:
        Decorated function with metadata for PythoC compilation
    """
    # Store metadata on the function object
    fn.__aie_kernel__ = True
    fn.__aie_source__ = inspect.getsource(fn)
    fn.__aie_name__ = fn.__name__
    fn.__aie_signature__ = inspect.signature(fn)
    
    return fn

# types.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

"""Type mapping utilities for PythoC-IRON integration.

Maps PythoC type annotations to NumPy types used by IRON.
"""

import re
import inspect
import numpy as np
from typing import Any


# Type mapping from PythoC type names to NumPy dtypes
PYTHOC_TO_NUMPY_MAP = {
    # Integer types
    'i8': np.int8,
    'i16': np.int16,
    'i32': np.int32,
    'i64': np.int64,
    'u8': np.uint8,
    'u16': np.uint16,
    'u32': np.uint32,
    'u64': np.uint64,
    
    # Floating point types
    'f16': np.float16,
    'f32': np.float32,
    'f64': np.float64,
    'bf16': np.dtype('bfloat16'),  # BFloat16
    
    # Boolean
    'bool': np.bool_,
}


def pythoc_to_numpy_type(pythoc_type_str: str) -> np.dtype:
    """Convert PythoC type string to NumPy dtype.
    
    Args:
        pythoc_type_str: PythoC type as string (e.g., 'i32', 'bf16', 'ptr[i32]')
        
    Returns:
        NumPy dtype
        
    Raises:
        ValueError: If type cannot be mapped
    """
    # Handle simple scalar types
    if pythoc_type_str in PYTHOC_TO_NUMPY_MAP:
        return np.dtype(PYTHOC_TO_NUMPY_MAP[pythoc_type_str])
    
    # Handle pointer types: ptr[T] or ptr[T, True]
    ptr_match = re.match(r'ptr\[(\w+)(?:,\s*(?:True|False))?\]', pythoc_type_str)
    if ptr_match:
        element_type = ptr_match.group(1)
        if element_type in PYTHOC_TO_NUMPY_MAP:
            return np.dtype(PYTHOC_TO_NUMPY_MAP[element_type])
        raise ValueError(f"Unknown pointer element type: {element_type}")
    
    # Handle vector types: aie_vector[T, N]
    vec_match = re.match(r'aie_vector\[(\w+),\s*(\d+)\]', pythoc_type_str)
    if vec_match:
        element_type = vec_match.group(1)
        # For vectors, return the element dtype (IRON will handle shape)
        if element_type in PYTHOC_TO_NUMPY_MAP:
            return np.dtype(PYTHOC_TO_NUMPY_MAP[element_type])
        raise ValueError(f"Unknown vector element type: {element_type}")
    
    raise ValueError(f"Cannot map PythoC type to NumPy: {pythoc_type_str}")


def _extract_pythoc_type_from_annotation(annotation: Any) -> str:
    """Extract PythoC type string from Python annotation object.
    
    Args:
        annotation: Python type annotation
        
    Returns:
        PythoC type string (e.g., 'i32', 'ptr[bf16, True]')
    """
    # Handle string annotations
    if isinstance(annotation, str):
        return annotation
    
    # Handle type objects with __name__
    if hasattr(annotation, '__name__'):
        return annotation.__name__
    
    # Handle subscripted types (ptr[T], aie_vector[T, N])
    if hasattr(annotation, '__origin__'):
        origin_name = getattr(annotation.__origin__, '__name__', str(annotation.__origin__))
        
        # Get args
        if hasattr(annotation, '__args__'):
            args = annotation.__args__
            if origin_name == 'ptr':
                # ptr[T] or ptr[T, True]
                if len(args) == 1:
                    elem_type = _extract_pythoc_type_from_annotation(args[0])
                    return f"ptr[{elem_type}]"
                elif len(args) == 2:
                    elem_type = _extract_pythoc_type_from_annotation(args[0])
                    restrict = args[1]
                    return f"ptr[{elem_type}, {restrict}]"
            elif origin_name == 'aie_vector':
                # aie_vector[T, N]
                if len(args) == 2:
                    elem_type = _extract_pythoc_type_from_annotation(args[0])
                    size = args[1]
                    return f"aie_vector[{elem_type}, {size}]"
    
    # Fallback: convert to string
    return str(annotation)


def infer_kernel_signature(pythoc_function: Any) -> list[type[np.ndarray] | np.dtype]:
    """Infer IRON kernel signature from PythoC function annotations.
    
    Extracts type annotations from a PythoC function and converts them
    to NumPy types suitable for IRON Kernel declarations.
    
    Args:
        pythoc_function: PythoC function with type annotations
        
    Returns:
        List of NumPy array types or dtypes for IRON Kernel
        
    Example:
        @compile
        def add(a: ptr[i32, True], b: ptr[i32, True], c: ptr[i32, True], N: i32):
            ...
        
        signature = infer_kernel_signature(add)
        # Returns: [
        #     np.ndarray[(...,), np.dtype[np.int32]],
        #     np.ndarray[(...,), np.dtype[np.int32]],
        #     np.ndarray[(...,), np.dtype[np.int32]],
        #     np.int32
        # ]
    """
    sig = inspect.signature(pythoc_function)
    iron_types = []
    
    for param_name, param in sig.parameters.items():
        if param.annotation == inspect.Parameter.empty:
            raise ValueError(
                f"Parameter '{param_name}' has no type annotation. "
                f"All PythoC kernel parameters must be typed."
            )
        
        # Extract PythoC type string
        pythoc_type_str = _extract_pythoc_type_from_annotation(param.annotation)
        
        # Convert to NumPy type
        numpy_dtype = pythoc_to_numpy_type(pythoc_type_str)
        
        # Determine if this should be an array or scalar
        if pythoc_type_str.startswith('ptr[') or pythoc_type_str.startswith('aie_vector['):
            # Pointer or vector → NumPy array with unknown shape
            iron_types.append(np.ndarray[(...,), np.dtype[numpy_dtype.type]])
        else:
            # Scalar → NumPy dtype
            iron_types.append(numpy_dtype.type)
    
    return iron_types

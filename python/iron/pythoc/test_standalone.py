#!/usr/bin/env python3
# test_standalone.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

"""Standalone test for PythoC compiler (no IRON dependencies)."""

import sys
import os
from pathlib import Path

# Add PythoC to path
pythoc_path = Path(__file__).parent.parent.parent.parent.parent / "PythoC"
sys.path.insert(0, str(pythoc_path.parent))

# Import compiler directly
from compiler import compile_pythoc_kernel


def test_compile_add_kernel():
    """Test compiling the mul.py kernel (no event markers)."""
    print("=" * 60)
    print("Test: Compile PythoC mul.py kernel to AIE2 object file")
    print("=" * 60 + "\n")
    
    # Path to PythoC kernel (use mul.py which has no event markers)
    kernel_path = pythoc_path / "pythoc_kernels" / "mul.py"
    
    if not kernel_path.exists():
        print(f"✗ SKIP: Kernel file not found: {kernel_path}")
        return False
    
    print(f"Kernel path: {kernel_path}")
    
    # Compile kernel
    try:
        obj_file = compile_pythoc_kernel(
            str(kernel_path),
            "eltwise_mul_vectorized_i32",
            target_arch="aie2",
            verbose=True
        )
        
        print(f"\n✓ Compilation successful!")
        print(f"✓ LLVM IR file: {obj_file}")
        print(f"✓ File exists: {obj_file.exists()}")
        
        # Verify IR file is not empty
        size = obj_file.stat().st_size
        print(f"✓ File size: {size} bytes")
        
        if size == 0:
            print("✗ FAIL: LLVM IR file is empty")
            return False
        
        # Check that it's a text file with LLVM IR
        with open(obj_file, 'r') as f:
            first_line = f.readline()
            if 'target triple' in first_line or 'define' in first_line:
                print("✓ Valid LLVM IR file")
            else:
                print(f"⚠ Warning: May not be valid LLVM IR (first line: {first_line[:50]})")
        
        print("\n" + "=" * 60)
        print("✓ PASS: All checks passed")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ FAIL: Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("✗ FAIL: Test failed")
        print("=" * 60)
        return False


if __name__ == "__main__":
    # Import the compiler module we just created
    compiler_module = Path(__file__).parent / "compiler.py"
    
    # Execute the module to get its functions
    import importlib.util
    spec = importlib.util.spec_from_file_location("compiler", compiler_module)
    compiler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiler)
    
    # Use the compile function
    compile_pythoc_kernel = compiler.compile_pythoc_kernel
    
    success = test_compile_add_kernel()
    sys.exit(0 if success else 1)

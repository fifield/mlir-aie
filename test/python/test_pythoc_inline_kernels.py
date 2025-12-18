#!/usr/bin/env python3
# test_phase2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

"""Test script for Phase 2: Inline PythoC kernel compilation."""

import sys
import tempfile
from pathlib import Path

# Test imports
try:
    from aie.iron.pythoc import aie_kernel, PythocKernel
    from pythoc import ptr, i32
    from pythoc.aie.operations import load_v, store_v, vector_add
    from pythoc.aie.vector import aie_vector
    import numpy as np
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


# Define a simple inline kernel
@aie_kernel
def test_add_kernel(a: ptr[i32, True], b: ptr[i32, True], 
                    c: ptr[i32, True], N: i32):
    """Simple vectorized addition kernel for testing."""
    vec_size: i32 = 16
    iterations: i32 = N // vec_size
    
    # Use pointer arithmetic (PythoC pattern)
    pA: ptr[i32] = a
    pB: ptr[i32] = b
    pC: ptr[i32] = c
    
    i: i32 = 0
    while i < iterations:
        va: aie_vector[i32, 16] = load_v(pA, 16)
        vb: aie_vector[i32, 16] = load_v(pB, 16)
        vc: aie_vector[i32, 16] = vector_add(va, vb)
        store_v(pC, vc)
        
        pA = pA + vec_size
        pB = pB + vec_size
        pC = pC + vec_size
        i = i + 1


def test_decorator():
    """Test that @aie_kernel decorator adds required metadata."""
    print("\n[Test 1] Decorator metadata...")
    
    assert hasattr(test_add_kernel, '__aie_kernel__'), "Missing __aie_kernel__ attribute"
    assert test_add_kernel.__aie_kernel__ == True, "__aie_kernel__ should be True"
    
    assert hasattr(test_add_kernel, '__aie_name__'), "Missing __aie_name__ attribute"
    assert test_add_kernel.__aie_name__ == 'test_add_kernel', "Incorrect function name"
    
    assert hasattr(test_add_kernel, '__aie_source__'), "Missing __aie_source__ attribute"
    assert 'def test_add_kernel' in test_add_kernel.__aie_source__, "Source code not captured"
    
    assert hasattr(test_add_kernel, '__aie_signature__'), "Missing __aie_signature__ attribute"
    
    print("  ✓ Decorator adds all required metadata")
    return True


def test_kernel_compilation():
    """Test that PythocKernel can compile inline decorated functions."""
    print("\n[Test 2] Inline kernel compilation...")
    
    try:
        # Define tile type
        tile_ty = np.ndarray[(256,), np.dtype[np.int32]]
        
        # Create PythocKernel from decorated function
        kernel = PythocKernel(
            test_add_kernel,
            [tile_ty, tile_ty, tile_ty, np.int32]
        )
        
        print(f"  ✓ Kernel compiled successfully")
        print(f"  ✓ Kernel name: {kernel._name}")
        print(f"  ✓ LLVM IR file: {kernel.bin_name}")
        
        # Verify LLVM IR file was created
        ll_file = Path(kernel.bin_name)
        if not ll_file.exists():
            print(f"  ✗ LLVM IR file not found: {ll_file}")
            return False
        
        # Check file size
        file_size = ll_file.stat().st_size
        print(f"  ✓ LLVM IR file size: {file_size} bytes")
        
        if file_size < 100:
            print(f"  ✗ LLVM IR file seems too small")
            return False
        
        # Read and check IR content
        with open(ll_file, 'r') as f:
            ir_content = f.read()
        
        # Verify function definition exists
        if f'define void @test_add_kernel' not in ir_content:
            print(f"  ✗ Function definition not found in IR")
            return False
        
        print(f"  ✓ LLVM IR contains function definition")
        
        # Check for vector operations
        if 'vector' in ir_content.lower() or '<16 x i32>' in ir_content:
            print(f"  ✓ LLVM IR contains vector operations")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n[Test 3] Error handling...")
    
    # Test 1: Non-decorated function
    def regular_function():
        pass
    
    try:
        tile_ty = np.ndarray[(256,), np.dtype[np.int32]]
        kernel = PythocKernel(regular_function, [tile_ty])
        print("  ✗ Should have raised TypeError for non-decorated function")
        return False
    except TypeError as e:
        if '@aie_kernel' in str(e):
            print("  ✓ Correctly rejects non-decorated function")
        else:
            print(f"  ✗ Wrong error message: {e}")
            return False
    
    # Test 2: Invalid types parameter
    try:
        kernel = PythocKernel(test_add_kernel, "not a list")
        print("  ✗ Should have raised TypeError for invalid types")
        return False
    except TypeError as e:
        if 'list of types' in str(e):
            print("  ✓ Correctly rejects invalid types parameter")
        else:
            print(f"  ✗ Wrong error message: {e}")
            return False
    
    return True


def main():
    """Run all Phase 2 tests."""
    print("=" * 60)
    print("PythoC IRON Integration - Phase 2 Tests")
    print("=" * 60)
    
    tests = [
        ("Decorator Metadata", test_decorator),
        ("Inline Kernel Compilation", test_kernel_compilation),
        ("Error Handling", test_error_handling),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All Phase 2 tests PASSED!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

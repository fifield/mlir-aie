#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

import argparse
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../python"))

import torch

import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime


def bf16_to_uint16(bf16_tensor):
    """Convert bfloat16 tensor to uint16 array for AIE."""
    return bf16_tensor.view(torch.uint16).cpu().numpy()


def uint16_to_bf16(uint16_array):
    """Convert uint16 array from AIE to bfloat16 tensor."""
    return torch.from_numpy(uint16_array.copy()).view(torch.bfloat16)


def test_elementwise(
    size,
    operation="add",
    use_aie=False,
    xclbin_path=None,
    insts_path=None,
):
    """Test element-wise operation."""
    
    print(f"\n{'='*80}")
    print(f"Testing Element-wise {operation.upper()} Operation:")
    print(f"  Size: {size} elements")
    print(f"{'='*80}\n")
    
    # Generate random inputs
    torch.manual_seed(42)
    a_tensor = torch.randn(size).to(torch.bfloat16)
    b_tensor = torch.randn(size).to(torch.bfloat16)
    
    # Compute PyTorch reference
    with torch.no_grad():
        if operation == "add":
            torch_output = a_tensor + b_tensor
        elif operation == "max":
            torch_output = torch.maximum(a_tensor, b_tensor)
        elif operation == "mul":
            torch_output = a_tensor * b_tensor
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    print(f"PyTorch output range: [{torch_output.min():.4f}, {torch_output.max():.4f}]")
    print(f"PyTorch output mean: {torch_output.mean():.4f}, std: {torch_output.std():.4f}")
    
    if use_aie:
        if xclbin_path is None or insts_path is None:
            print("ERROR: xclbin_path and insts_path required")
            return False
        
        print(f"\n{'='*80}")
        print("Running on NPU2 Hardware")
        print(f"{'='*80}\n")
        
        # Convert to uint16
        a_uint16 = bf16_to_uint16(a_tensor)
        b_uint16 = bf16_to_uint16(b_tensor)
        
        print(f"Buffer sizes: {size} elements ({size * 2} bytes each)")

        # Setup NPU kernel
        npu_kernel = NPUKernel(xclbin_path, insts_path, kernel_name="MLIR_AIE")
        kernel_handle = DefaultNPURuntime.load(npu_kernel)

        in1 = iron.tensor(a_uint16, dtype=np.uint16)
        in2 = iron.tensor(b_uint16, dtype=np.uint16)
        out = iron.zeros(size, dtype=np.uint16)

        start = time.time_ns()
        ret = DefaultNPURuntime.run(kernel_handle, [in1, in2, out])
        stop = time.time_ns()
        npu_time = (stop - start) / 1000

        print(f"Execution time: {npu_time:.2f} μs ({npu_time/1000:.3f} ms)")

        # Convert back
        aie_output = uint16_to_bf16(out.numpy()[:size])
        
        print(f"AIE output range: [{aie_output.min():.4f}, {aie_output.max():.4f}]")
        print(f"AIE output mean: {aie_output.mean():.4f}, std: {aie_output.std():.4f}")
        
        # Compare
        diff = torch.abs(torch_output - aie_output)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"\nComparison (PyTorch vs AIE):")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        
        tolerance = 0.05  # Relaxed for bfloat16 precision
        if max_diff < tolerance:
            print(f"  ✓ PASS (max diff < {tolerance})")
            return True
        else:
            print(f"  ✗ FAIL (max diff >= {tolerance})")
            return False
    else:
        # CPU reference
        print("\nRunning CPU reference (NumPy)...")
        
        a_np = a_tensor.float().numpy()
        b_np = b_tensor.float().numpy()
        
        if operation == "add":
            output_np = a_np + b_np
        elif operation == "max":
            output_np = np.maximum(a_np, b_np)
        elif operation == "mul":
            output_np = a_np * b_np
        
        output_tensor = torch.from_numpy(output_np)
        
        print(f"NumPy output range: [{output_tensor.min():.4f}, {output_tensor.max():.4f}]")
        print(f"NumPy output mean: {output_tensor.mean():.4f}, std: {output_tensor.std():.4f}")
        
        diff = torch.abs(torch_output - output_tensor)
        max_diff = diff.max().item()
        
        print(f"\nComparison: Max diff = {max_diff:.6f}")
        
        tolerance = 0.05  # Relaxed for bfloat16 precision
        if max_diff < tolerance:
            print(f"  ✓ PASS (max diff < {tolerance})")
            return True
        else:
            print(f"  ✗ FAIL (max diff >= {tolerance})")
            return False


def main():
    parser = argparse.ArgumentParser(description="Test element-wise operations")
    parser.add_argument("--size", "-s", type=int, default=512, help="Tensor size")
    parser.add_argument("--operation", "-op", type=str, default="add", 
                       choices=["add", "max", "mul"], help="Operation type")
    parser.add_argument("--xclbin", "-x", type=str, help="Path to xclbin file")
    parser.add_argument("--insts", "-i", type=str, help="Path to instructions file")
    
    args = parser.parse_args()
    use_aie = args.xclbin is not None and args.insts is not None
    
    success = test_elementwise(
        args.size, args.operation, use_aie,
        args.xclbin, args.insts
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

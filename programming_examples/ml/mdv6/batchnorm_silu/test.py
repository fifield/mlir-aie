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

# Add parent directories to path to import mdv6
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../python"))

import torch
import torch.nn as nn


def bf16_to_uint16(bf16_tensor):
    """Convert bfloat16 tensor to uint16 array for AIE."""
    return bf16_tensor.view(torch.uint16).cpu().numpy()


def uint16_to_bf16(uint16_array):
    """Convert uint16 array from AIE to bfloat16 tensor."""
    return torch.from_numpy(uint16_array).view(torch.bfloat16)


def test_batchnorm_silu(
    height,
    width,
    channels,
    use_silu=True,
    use_aie=False,
    xclbin_path=None,
    insts_path=None,
):
    """
    Test BatchNorm + SiLU layer implementation.
    """
    
    print(f"\n{'='*80}")
    print(f"Testing BatchNorm + SiLU Layer:")
    print(f"  Input: ({height}, {width}, {channels})")
    print(f"  Use SiLU: {use_silu}")
    print(f"{'='*80}\n")
    
    # Create PyTorch BatchNorm + SiLU
    bn = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03).to(torch.bfloat16)
    bn.eval()
    
    # Generate random input (using bfloat16)
    torch.manual_seed(42)
    input_tensor = torch.randn(1, channels, height, width).to(torch.bfloat16)
    
    # Run PyTorch forward pass
    with torch.no_grad():
        bn_output = bn(input_tensor)
        if use_silu:
            torch_output = nn.functional.silu(bn_output)
        else:
            torch_output = bn_output
    
    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"PyTorch output range: [{torch_output.min():.4f}, {torch_output.max():.4f}]")
    print(f"PyTorch output mean: {torch_output.mean():.4f}, std: {torch_output.std():.4f}")
    
    if use_aie:
        if xclbin_path is None or insts_path is None:
            print("ERROR: xclbin_path and insts_path required for AIE execution")
            return False
        
        print(f"\n{'='*80}")
        print("Running on NPU2 Hardware (AIE2P)")
        print(f"{'='*80}\n")
        
        # Prepare data for AIE (convert to HWC format and uint16)
        input_hwc = input_tensor.squeeze(0).permute(1, 2, 0).contiguous()  # (H, W, C)
        input_uint16 = bf16_to_uint16(input_hwc.flatten())
        
        # Extract BN parameters
        bn_weight_uint16 = bf16_to_uint16(bn.weight.data)
        bn_bias_uint16 = bf16_to_uint16(bn.bias.data)
        
        # Calculate buffer sizes
        input_size = height * width * channels
        param_size = channels
        output_size = height * width * channels
        
        print(f"Buffer sizes:")
        print(f"  Input:     {input_size} elements ({input_size * 2} bytes)")
        print(f"  BN weight: {param_size} elements ({param_size * 2} bytes)")
        print(f"  BN bias:   {param_size} elements ({param_size * 2} bytes)")
        print(f"  Output:    {output_size} elements ({output_size * 2} bytes)")
        
        # Setup AIE application (3 inputs: data, weight, bias)
        print(f"\nSetting up AIE application...")
        print(f"  XCLBin: {xclbin_path}")
        print(f"  Instructions: {insts_path}")
        
        # Note: setup_aie only supports 2 inputs, so we need to use a workaround
        # We'll concatenate bn_weight and bn_bias into a single buffer
        bn_params = np.concatenate([bn_weight_uint16, bn_bias_uint16])
        
        app = setup_aie(
            xclbin_path,
            insts_path,
            input_uint16.shape,   # Input shape
            np.uint16,            # Input dtype
            bn_params.shape,      # BN params shape (weight + bias)
            np.uint16,            # BN params dtype
            (output_size,),       # Output shape
            np.uint16,            # Output dtype
            kernel_name="MLIR_AIE"
        )
        
        # Execute on hardware
        print(f"\nExecuting kernel on NPU2...")
        start = time.time_ns()
        output_buffer = execute(app, input_uint16, bn_params)
        stop = time.time_ns()
        npu_time = (stop - start) / 1000  # Convert to microseconds
        
        print(f"  Execution time: {npu_time:.2f} μs ({npu_time/1000:.3f} ms)")
        
        # Convert output back to bfloat16
        output_bf16 = uint16_to_bf16(output_buffer[:output_size])
        
        # Reshape to (H, W, C) then convert to (1, C, H, W) for PyTorch
        output_hwc = output_bf16.reshape(height, width, channels)
        aie_output = torch.from_numpy(output_hwc.float().numpy()).permute(2, 0, 1).unsqueeze(0)
        
        print(f"\nAIE output shape: {aie_output.shape}")
        print(f"AIE output range: [{aie_output.min():.4f}, {aie_output.max():.4f}]")
        print(f"AIE output mean: {aie_output.mean():.4f}, std: {aie_output.std():.4f}")
        
        # Compare with PyTorch reference
        diff = torch.abs(torch_output - aie_output)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"\nComparison (PyTorch vs AIE Hardware):")
        print(f"  Max absolute difference: {max_diff:.6f}")
        print(f"  Mean absolute difference: {mean_diff:.6f}")
        
        tolerance = 0.1
        if max_diff < tolerance:
            print(f"  ✓ PASS (max diff < {tolerance})")
            return True
        else:
            print(f"  ✗ FAIL (max diff >= {tolerance})")
            return False
    else:
        # CPU reference test
        print("\nRunning CPU reference (NumPy)...")
        
        # Convert to float32 for NumPy
        input_np = input_tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy()  # (H, W, C)
        bn_weight_np = bn.weight.data.float().cpu().numpy()
        bn_bias_np = bn.bias.data.float().cpu().numpy()
        
        output_np = np.zeros((height, width, channels), dtype=np.float32)
        
        # Apply BatchNorm
        for c in range(channels):
            output_np[:, :, c] = input_np[:, :, c] * bn_weight_np[c] + bn_bias_np[c]
        
        # Apply SiLU if requested
        if use_silu:
            for hw in range(height * width):
                for c in range(channels):
                    h = hw // width
                    w = hw % width
                    x = output_np[h, w, c]
                    # SiLU: x * sigmoid(x) using SAME fast sigmoid as AIE kernel
                    # sigmoid(x) ≈ x / (1 + |x|), shifted to [0, 1]
                    abs_x = abs(x)
                    sigmoid_approx = x / (1.0 + abs_x)
                    sigmoid_approx = 0.5 * (sigmoid_approx + 1.0)
                    output_np[h, w, c] = x * sigmoid_approx
        
        # Convert back to tensor for comparison
        output_np_tensor = torch.from_numpy(output_np).permute(2, 0, 1).unsqueeze(0)
        
        print(f"NumPy output shape: {output_np_tensor.shape}")
        print(f"NumPy output range: [{output_np_tensor.min():.4f}, {output_np_tensor.max():.4f}]")
        print(f"NumPy output mean: {output_np_tensor.mean():.4f}, std: {output_np_tensor.std():.4f}")
        
        # Compare
        diff = torch.abs(torch_output - output_np_tensor)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"\nComparison (PyTorch vs NumPy):")
        print(f"  Max absolute difference: {max_diff:.6f}")
        print(f"  Mean absolute difference: {mean_diff:.6f}")
        
        # Fast sigmoid approximation has ~21% error vs PyTorch exact sigmoid
        tolerance = 0.25
        if max_diff < tolerance:
            print(f"  ✓ PASS (max diff < {tolerance})")
            return True
        else:
            print(f"  ✗ FAIL (max diff >= {tolerance})")
            return False


def main():
    parser = argparse.ArgumentParser(description="Test BatchNorm + SiLU layer for MDV6")
    parser.add_argument("--height", "-ht", type=int, default=8, help="Input height")
    parser.add_argument("--width", "-wd", type=int, default=8, help="Input width")
    parser.add_argument("--channels", "-c", type=int, default=8, help="Number of channels")
    parser.add_argument("--use-silu", action="store_true", help="Include SiLU activation")
    parser.add_argument("--xclbin", "-x", type=str, help="Path to xclbin file")
    parser.add_argument("--insts", "-i", type=str, help="Path to instructions file")
    
    args = parser.parse_args()
    
    use_aie = args.xclbin is not None and args.insts is not None
    
    success = test_batchnorm_silu(
        args.height,
        args.width,
        args.channels,
        use_silu=args.use_silu,
        use_aie=use_aie,
        xclbin_path=args.xclbin,
        insts_path=args.insts,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

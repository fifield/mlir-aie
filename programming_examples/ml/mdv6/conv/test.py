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
from mdv6.layers import Conv


def bf16_to_uint16(bf16_tensor):
    """Convert bfloat16 tensor to uint16 array for AIE."""
    return bf16_tensor.view(torch.uint16).cpu().numpy()


def uint16_to_bf16(uint16_array):
    """Convert uint16 array from AIE to bfloat16 tensor."""
    return torch.from_numpy(uint16_array).view(torch.bfloat16)


def test_conv_layer(
    input_height,
    input_width,
    input_channels,
    output_channels,
    kernel_size,
    stride,
    padding,
    use_aie=False,
    xclbin_path=None,
    insts_path=None,
):
    """
    Test Conv layer implementation.
    
    Args:
        input_height: Input height
        input_width: Input width
        input_channels: Number of input channels
        output_channels: Number of output channels
        kernel_size: Kernel size (1 or 3)
        stride: Stride
        padding: Padding
        use_aie: Whether to run on AIE hardware
        xclbin_path: Path to xclbin file
        insts_path: Path to instructions file
    """
    
    print(f"\n{'='*80}")
    print(f"Testing Conv Layer:")
    print(f"  Input: ({input_height}, {input_width}, {input_channels})")
    print(f"  Output channels: {output_channels}")
    print(f"  Kernel size: {kernel_size}x{kernel_size}")
    print(f"  Stride: {stride}, Padding: {padding}")
    print(f"{'='*80}\n")
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    
    # Create PyTorch Conv layer
    torch_conv = Conv(
        input_channels,
        output_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        activation="SiLU"  # Note: Our AIE kernel doesn't include activation yet
    )
    torch_conv.eval()
    
    # Convert model to bfloat16
    torch_conv = torch_conv.to(torch.bfloat16)
    
    # Generate random input (using bfloat16)
    torch.manual_seed(42)
    input_tensor = torch.randn(1, input_channels, input_height, input_width).to(torch.bfloat16)
    
    # Run PyTorch forward pass
    with torch.no_grad():
        # Get conv output before activation
        conv_out = torch_conv.bn(torch_conv.conv(input_tensor))
        # For now, compare without activation since AIE kernel doesn't have it yet
        torch_output = conv_out
    
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
        # PyTorch uses NCHW, AIE kernel expects HWC
        input_hwc = input_tensor.squeeze(0).permute(1, 2, 0).contiguous()  # (H, W, C)
        input_uint16 = bf16_to_uint16(input_hwc.flatten())
        
        # Extract weights (convert to OIHW format for kernel)
        weights = torch_conv.conv.weight.data  # (O, I, K, K)
        weights_uint16 = bf16_to_uint16(weights.flatten())
        
        # Calculate buffer sizes
        input_size = input_height * input_width * input_channels
        weight_size = output_channels * input_channels * kernel_size * kernel_size
        output_size = output_height * output_width * output_channels
        
        print(f"Buffer sizes:")
        print(f"  Input:   {input_size} elements ({input_size * 2} bytes)")
        print(f"  Weights: {weight_size} elements ({weight_size * 2} bytes)")
        print(f"  Output:  {output_size} elements ({output_size * 2} bytes)")
        
        # Setup AIE application
        print(f"\nSetting up AIE application...")
        print(f"  XCLBin: {xclbin_path}")
        print(f"  Instructions: {insts_path}")
        
        app = setup_aie(
            xclbin_path,
            insts_path,
            input_uint16.shape,   # Input shape
            np.uint16,            # Input dtype (bf16 as uint16)
            weights_uint16.shape, # Weights shape
            np.uint16,            # Weights dtype
            (output_size,),       # Output shape
            np.uint16,            # Output dtype
            kernel_name="MLIR_AIE"
        )
        
        # Execute on hardware
        print(f"\nExecuting kernel on NPU2...")
        start = time.time_ns()
        output_buffer = execute(app, input_uint16, weights_uint16)
        stop = time.time_ns()
        npu_time = (stop - start) / 1000  # Convert to microseconds
        
        print(f"  Execution time: {npu_time:.2f} μs ({npu_time/1000:.3f} ms)")
        
        # Convert output back to bfloat16
        output_bf16 = uint16_to_bf16(output_buffer[:output_size])
        
        # Reshape to (H, W, C) then convert to (1, C, H, W) for PyTorch
        output_hwc = output_bf16.reshape(output_height, output_width, output_channels)
        # Convert to float32 for NumPy compatibility, then back to tensor
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
        
        # Check if close enough (bfloat16 has limited precision)
        tolerance = 0.1  # Relaxed tolerance for bfloat16
        if max_diff < tolerance:
            print(f"  ✓ PASS (max diff < {tolerance})")
            return True
        else:
            print(f"  ✗ FAIL (max diff >= {tolerance})")
            return False
    else:
        # CPU reference test
        print("\nRunning CPU reference (NumPy)...")
        
        # Simple reference implementation (convert to float32 for NumPy)
        input_np = input_tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy()  # (H, W, C)
        weights_np = torch_conv.conv.weight.data.float().cpu().numpy()  # (O, I, K, K)
        
        output_np = np.zeros((output_height, output_width, output_channels), dtype=np.float32)
        
        # Simple convolution (very slow, just for validation)
        for oc in range(output_channels):
            for oh in range(output_height):
                for ow in range(output_width):
                    sum_val = 0.0
                    for ic in range(input_channels):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                ih = oh * stride + kh - padding
                                iw = ow * stride + kw - padding
                                if 0 <= ih < input_height and 0 <= iw < input_width:
                                    sum_val += float(input_np[ih, iw, ic]) * float(weights_np[oc, ic, kh, kw])
                    output_np[oh, ow, oc] = sum_val
        
        # Apply BatchNorm (simplified - just scale and bias)
        bn_weight = torch_conv.bn.weight.data.float().cpu().numpy()
        bn_bias = torch_conv.bn.bias.data.float().cpu().numpy()
        for oc in range(output_channels):
            output_np[:, :, oc] = output_np[:, :, oc] * bn_weight[oc] + bn_bias[oc]
        
        # Convert back to tensor for comparison
        output_np_tensor = torch.from_numpy(output_np).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        
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
        
        # Check if close enough (bfloat16 has limited precision)
        tolerance = 0.1  # Relaxed tolerance for bfloat16
        if max_diff < tolerance:
            print(f"  ✓ PASS (max diff < {tolerance})")
            return True
        else:
            print(f"  ✗ FAIL (max diff >= {tolerance})")
            return False


def main():
    parser = argparse.ArgumentParser(description="Test Conv layer for MDV6")
    parser.add_argument("--height", "-ht", type=int, default=8, help="Input height")
    parser.add_argument("--width", "-wd", type=int, default=8, help="Input width")
    parser.add_argument("--in-channels", "-ic", type=int, default=8, help="Input channels")
    parser.add_argument("--out-channels", "-oc", type=int, default=8, help="Output channels")
    parser.add_argument("--kernel-size", "-k", type=int, default=3, help="Kernel size")
    parser.add_argument("--stride", "-s", type=int, default=1, help="Stride")
    parser.add_argument("--padding", "-p", type=int, default=1, help="Padding")
    parser.add_argument("--xclbin", "-x", type=str, help="Path to xclbin file")
    parser.add_argument("--insts", "-i", type=str, help="Path to instructions file")
    
    args = parser.parse_args()
    
    use_aie = args.xclbin is not None and args.insts is not None
    
    success = test_conv_layer(
        args.height,
        args.width,
        args.in_channels,
        args.out_channels,
        args.kernel_size,
        args.stride,
        args.padding,
        use_aie=use_aie,
        xclbin_path=args.xclbin,
        insts_path=args.insts,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

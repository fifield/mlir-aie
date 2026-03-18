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
from mdv6.layers import Bottleneck

import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime


def bf16_to_uint16(bf16_tensor):
    """Convert bfloat16 tensor to uint16 array for AIE."""
    return bf16_tensor.view(torch.uint16).cpu().numpy()


def uint16_to_bf16(uint16_array):
    """Convert uint16 array from AIE to bfloat16 tensor."""
    return torch.from_numpy(uint16_array.copy()).view(torch.bfloat16)


def test_bottleneck_layer(
    height,
    width,
    in_channels,
    out_channels,
    stride=1,
    padding=1,
    residual=True,
    use_aie=False,
    xclbin_path=None,
    insts_path=None,
):
    """
    Test Bottleneck layer implementation.
    
    Bottleneck = RepConv → Conv+BN+SiLU → optional residual add
    
    Args:
        height: Input height
        width: Input width
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for convolutions
        padding: Padding for convolutions
        residual: Enable residual connection
        use_aie: Whether to run on AIE hardware
        xclbin_path: Path to xclbin file
        insts_path: Path to instructions file
    """
    
    print(f"\n{'='*80}")
    print(f"Testing Bottleneck Layer:")
    print(f"  Input: ({height}, {width}, {in_channels})")
    print(f"  Output channels: {out_channels}")
    print(f"  Stride: {stride}, Padding: {padding}")
    print(f"  Residual: {residual}")
    print(f"  Architecture: RepConv → Conv+BN+SiLU → {'Add (residual)' if residual else 'No residual'}")
    print(f"{'='*80}\n")
    
    # Calculate output dimensions
    output_height = (height + 2 * padding - 3) // stride + 1
    output_width = (width + 2 * padding - 3) // stride + 1
    
    print(f"Output dimensions: ({output_height}, {output_width}, {out_channels})")
    
    # Create PyTorch Bottleneck layer
    torch_bottleneck = Bottleneck(in_channels, out_channels, 
                                   kernel_size=(3, 3), 
                                   residual=residual, 
                                   expand=1.0)
    torch_bottleneck.eval()
    
    # Convert model to bfloat16
    torch_bottleneck = torch_bottleneck.to(torch.bfloat16)
    
    # Generate random input (using bfloat16)
    torch.manual_seed(42)
    input_tensor = torch.randn(1, in_channels, height, width).to(torch.bfloat16)
    
    # Run PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_bottleneck(input_tensor)
    
    print(f"\nPyTorch output shape: {torch_output.shape}")
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
        
        # Extract weights and BN parameters
        # RepConv (conv1) weights
        conv3x3_weights = torch_bottleneck.conv1.conv1.conv.weight.data  # (O, I, 3, 3)
        bn3x3_weight = torch_bottleneck.conv1.conv1.bn.weight.data
        bn3x3_bias = torch_bottleneck.conv1.conv1.bn.bias.data
        bn3x3_mean = torch_bottleneck.conv1.conv1.bn.running_mean.data
        bn3x3_var = torch_bottleneck.conv1.conv1.bn.running_var.data
        
        conv1x1_weights = torch_bottleneck.conv1.conv2.conv.weight.data  # (O, I, 1, 1)
        bn1x1_weight = torch_bottleneck.conv1.conv2.bn.weight.data
        bn1x1_bias = torch_bottleneck.conv1.conv2.bn.bias.data
        bn1x1_mean = torch_bottleneck.conv1.conv2.bn.running_mean.data
        bn1x1_var = torch_bottleneck.conv1.conv2.bn.running_var.data
        
        # Conv2 weights
        conv2_weights = torch_bottleneck.conv2.conv.weight.data  # (O, O, 3, 3)
        bn2_weight = torch_bottleneck.conv2.bn.weight.data
        bn2_bias = torch_bottleneck.conv2.bn.bias.data
        bn2_mean = torch_bottleneck.conv2.bn.running_mean.data
        bn2_var = torch_bottleneck.conv2.bn.running_var.data
        
        # Concatenate all weights and BN params
        weights_flat = torch.cat([
            conv3x3_weights.flatten(),
            bn3x3_weight, bn3x3_bias, bn3x3_mean, bn3x3_var,
            conv1x1_weights.flatten(),
            bn1x1_weight, bn1x1_bias, bn1x1_mean, bn1x1_var,
            conv2_weights.flatten(),
            bn2_weight, bn2_bias, bn2_mean, bn2_var,
        ])
        weights_uint16 = bf16_to_uint16(weights_flat)
        
        # Calculate buffer sizes
        input_size = height * width * in_channels
        output_size = output_height * output_width * out_channels
        
        print(f"Buffer sizes:")
        print(f"  Input:     {input_size} elements ({input_size * 2} bytes)")
        print(f"  Weights:   {len(weights_uint16)} elements ({len(weights_uint16) * 2} bytes)")
        print(f"  Output:    {output_size} elements ({output_size * 2} bytes)")
        
        # Setup NPU kernel
        print(f"\nSetting up AIE application...")
        print(f"  XCLBin: {xclbin_path}")
        print(f"  Instructions: {insts_path}")

        npu_kernel = NPUKernel(xclbin_path, insts_path, kernel_name="MLIR_AIE")
        kernel_handle = DefaultNPURuntime.load(npu_kernel)

        in1 = iron.tensor(input_uint16, dtype=np.uint16)
        in2 = iron.tensor(weights_uint16, dtype=np.uint16)
        out = iron.zeros(output_size, dtype=np.uint16)

        # Execute on hardware
        print(f"\nExecuting kernel on NPU2...")
        start = time.time_ns()
        ret = DefaultNPURuntime.run(kernel_handle, [in1, in2, out])
        stop = time.time_ns()
        npu_time = (stop - start) / 1000  # Convert to microseconds

        print(f"  Execution time: {npu_time:.2f} μs ({npu_time/1000:.3f} ms)")

        # Convert output back to bfloat16
        output_bf16 = uint16_to_bf16(out.numpy()[:output_size])
        
        # Reshape to (H, W, C) then convert to (1, C, H, W) for PyTorch
        output_hwc = output_bf16.reshape(output_height, output_width, out_channels)
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
        
        # Check if close enough (higher tolerance due to multiple approximations)
        tolerance = 0.35  # 3× sqrt + 2× sigmoid approximations
        if max_diff < tolerance:
            print(f"  ✓ PASS (max diff < {tolerance})")
            return True
        else:
            print(f"  ✗ FAIL (max diff >= {tolerance})")
            print(f"\nNote: Error is due to cumulative approximations:")
            print(f"  - 3× sqrt approximation (3 BatchNorms)")
            print(f"  - 2× sigmoid approximation (2 SiLU activations)")
            return False
    else:
        # CPU reference test - just verify PyTorch works
        print("\n✓ PyTorch reference test complete")
        return True


def main():
    parser = argparse.ArgumentParser(description="Test Bottleneck layer for MDV6")
    parser.add_argument("--height", "-ht", type=int, default=8, help="Input height")
    parser.add_argument("--width", "-wd", type=int, default=8, help="Input width")
    parser.add_argument("--in-channels", "-ic", type=int, default=8, help="Input channels")
    parser.add_argument("--out-channels", "-oc", type=int, default=8, help="Output channels")
    parser.add_argument("--stride", "-s", type=int, default=1, help="Stride")
    parser.add_argument("--padding", "-p", type=int, default=1, help="Padding")
    parser.add_argument("--residual", "-r", type=int, default=1, help="Enable residual (0 or 1)")
    parser.add_argument("--xclbin", "-x", type=str, help="Path to xclbin file")
    parser.add_argument("--insts", "-i", type=str, help="Path to instructions file")
    
    args = parser.parse_args()
    
    use_aie = args.xclbin is not None and args.insts is not None
    
    success = test_bottleneck_layer(
        args.height,
        args.width,
        args.in_channels,
        args.out_channels,
        args.stride,
        args.padding,
        bool(args.residual),
        use_aie=use_aie,
        xclbin_path=args.xclbin,
        insts_path=args.insts,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

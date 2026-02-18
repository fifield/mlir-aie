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
from mdv6.layers import AConv


def bf16_to_uint16(bf16_tensor):
    """Convert bfloat16 tensor to uint16 array for AIE."""
    return bf16_tensor.view(torch.uint16).cpu().numpy()


def uint16_to_bf16(uint16_array):
    """Convert uint16 array from AIE to bfloat16 tensor."""
    return torch.from_numpy(uint16_array).view(torch.bfloat16)


def test_aconv_layer(
    input_height,
    input_width,
    input_channels,
    output_channels,
    use_aie=False,
    xclbin_path=None,
    insts_path=None,
):
    """
    Test AConv layer implementation.
    
    AConv = AvgPool2d(kernel=2, stride=1, padding=0) + Conv3x3(stride=2, padding=1) + BN + SiLU
    
    Args:
        input_height: Input height
        input_width: Input width
        input_channels: Number of input channels
        output_channels: Number of output channels
        use_aie: Whether to run on AIE hardware
        xclbin_path: Path to xclbin file
        insts_path: Path to instructions file
    """
    
    print(f"\n{'='*80}")
    print(f"Testing AConv Layer:")
    print(f"  Input: ({input_height}, {input_width}, {input_channels})")
    print(f"  Output channels: {output_channels}")
    print(f"  Architecture: AvgPool2d(2×2, s=1) → Conv3x3(s=2, p=1) → BN → SiLU")
    print(f"{'='*80}\n")
    
    # Calculate output dimensions
    # After AvgPool: (H-1, W-1, C_in)
    pooled_height = input_height - 1
    pooled_width = input_width - 1
    # After Conv (stride=2, padding=1): ((H-1+2-3)/2+1, (W-1+2-3)/2+1, C_out)
    output_height = (pooled_height + 2 - 3) // 2 + 1
    output_width = (pooled_width + 2 - 3) // 2 + 1
    
    print(f"Dimension flow:")
    print(f"  Input:      ({input_height}, {input_width}, {input_channels})")
    print(f"  After pool: ({pooled_height}, {pooled_width}, {input_channels})")
    print(f"  Output:     ({output_height}, {output_width}, {output_channels})")
    
    # Create PyTorch AConv layer
    torch_aconv = AConv(input_channels, output_channels)
    torch_aconv.eval()
    
    # Convert model to bfloat16
    torch_aconv = torch_aconv.to(torch.bfloat16)
    
    # Generate random input (using bfloat16)
    torch.manual_seed(42)
    input_tensor = torch.randn(1, input_channels, input_height, input_width).to(torch.bfloat16)
    
    # Run PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_aconv(input_tensor)
    
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
        # PyTorch uses NCHW, AIE kernel expects HWC
        input_hwc = input_tensor.squeeze(0).permute(1, 2, 0).contiguous()  # (H, W, C)
        input_uint16 = bf16_to_uint16(input_hwc.flatten())
        
        # Extract weights and BN parameters
        # Conv weights: (O, I, K, K)
        conv_weights = torch_aconv.conv.conv.weight.data
        bn_weight = torch_aconv.conv.bn.weight.data
        bn_bias = torch_aconv.conv.bn.bias.data
        bn_mean = torch_aconv.conv.bn.running_mean.data
        bn_var = torch_aconv.conv.bn.running_var.data
        bn_eps = torch_aconv.conv.bn.eps
        
        # Fuse BN into conv weights (for reference comparison)
        # w_fused = w * (gamma / sqrt(var + eps))
        # b_fused = beta - (gamma * mean / sqrt(var + eps))
        bn_scale = bn_weight / torch.sqrt(bn_var + bn_eps)
        fused_weights = conv_weights * bn_scale.view(-1, 1, 1, 1)
        fused_bias = bn_bias - bn_scale * bn_mean
        
        # For AIE kernel, we pass weights and BN params separately
        # Concatenate: [weights, bn_weight, bn_bias, bn_mean, bn_var]
        weights_flat = conv_weights.flatten()
        bn_params = torch.cat([bn_weight, bn_bias, bn_mean, bn_var])
        weights_and_bn = torch.cat([weights_flat, bn_params])
        weights_uint16 = bf16_to_uint16(weights_and_bn)
        
        # Calculate buffer sizes
        input_size = input_height * input_width * input_channels
        weight_size = output_channels * input_channels * 3 * 3
        bn_param_size = 4 * output_channels  # weight, bias, mean, var
        total_weight_size = weight_size + bn_param_size
        output_size = output_height * output_width * output_channels
        
        print(f"Buffer sizes:")
        print(f"  Input:      {input_size} elements ({input_size * 2} bytes)")
        print(f"  Weights:    {weight_size} elements ({weight_size * 2} bytes)")
        print(f"  BN params:  {bn_param_size} elements ({bn_param_size * 2} bytes)")
        print(f"  Total wts:  {total_weight_size} elements ({total_weight_size * 2} bytes)")
        print(f"  Output:     {output_size} elements ({output_size * 2} bytes)")
        
        # Setup AIE application
        print(f"\nSetting up AIE application...")
        print(f"  XCLBin: {xclbin_path}")
        print(f"  Instructions: {insts_path}")
        
        app = setup_aie(
            xclbin_path,
            insts_path,
            input_uint16.shape,      # Input shape
            np.uint16,               # Input dtype (bf16 as uint16)
            weights_uint16.shape,    # Weights shape
            np.uint16,               # Weights dtype
            (output_size,),          # Output shape
            np.uint16,               # Output dtype
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
        
        # Check if close enough (bfloat16 has limited precision + sigmoid approximation)
        tolerance = 0.25  # Relaxed tolerance for bfloat16 + sigmoid approximation
        if max_diff < tolerance:
            print(f"  ✓ PASS (max diff < {tolerance})")
            return True
        else:
            print(f"  ✗ FAIL (max diff >= {tolerance})")
            print(f"\nNote: High error may be due to sigmoid approximation in SiLU")
            return False
    else:
        # CPU reference test
        print("\nRunning CPU reference (NumPy)...")
        
        # Manual implementation for validation
        input_np = input_tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy()  # (H, W, C)
        
        # Stage 1: AvgPool2d
        pooled_np = np.zeros((pooled_height, pooled_width, input_channels), dtype=np.float32)
        for oh in range(pooled_height):
            for ow in range(pooled_width):
                for c in range(input_channels):
                    sum_val = 0.0
                    for kh in range(2):
                        for kw in range(2):
                            sum_val += input_np[oh + kh, ow + kw, c]
                    pooled_np[oh, ow, c] = sum_val / 4.0
        
        # Stage 2: Conv3x3
        conv_weights_np = torch_aconv.conv.conv.weight.data.float().cpu().numpy()  # (O, I, K, K)
        conv_out_np = np.zeros((output_height, output_width, output_channels), dtype=np.float32)
        
        for oc in range(output_channels):
            for oh in range(output_height):
                for ow in range(output_width):
                    sum_val = 0.0
                    for ic in range(input_channels):
                        for kh in range(3):
                            for kw in range(3):
                                ih = oh * 2 + kh - 1  # stride=2, padding=1
                                iw = ow * 2 + kw - 1
                                if 0 <= ih < pooled_height and 0 <= iw < pooled_width:
                                    sum_val += pooled_np[ih, iw, ic] * conv_weights_np[oc, ic, kh, kw]
                    conv_out_np[oh, ow, oc] = sum_val
        
        # Stage 3: BatchNorm
        bn_weight_np = torch_aconv.conv.bn.weight.data.float().cpu().numpy()
        bn_bias_np = torch_aconv.conv.bn.bias.data.float().cpu().numpy()
        bn_mean_np = torch_aconv.conv.bn.running_mean.data.float().cpu().numpy()
        bn_var_np = torch_aconv.conv.bn.running_var.data.float().cpu().numpy()
        bn_eps = torch_aconv.conv.bn.eps
        
        for oc in range(output_channels):
            conv_out_np[:, :, oc] = (conv_out_np[:, :, oc] - bn_mean_np[oc]) / np.sqrt(bn_var_np[oc] + bn_eps)
            conv_out_np[:, :, oc] = conv_out_np[:, :, oc] * bn_weight_np[oc] + bn_bias_np[oc]
        
        # Stage 4: SiLU activation
        def silu(x):
            return x / (1.0 + np.exp(-x))
        
        output_np = silu(conv_out_np)
        
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
        
        # Check if close enough
        tolerance = 0.1
        if max_diff < tolerance:
            print(f"  ✓ PASS (max diff < {tolerance})")
            return True
        else:
            print(f"  ✗ FAIL (max diff >= {tolerance})")
            return False


def main():
    parser = argparse.ArgumentParser(description="Test AConv layer for MDV6")
    parser.add_argument("--height", "-ht", type=int, default=8, help="Input height")
    parser.add_argument("--width", "-wd", type=int, default=8, help="Input width")
    parser.add_argument("--in-channels", "-ic", type=int, default=8, help="Input channels")
    parser.add_argument("--out-channels", "-oc", type=int, default=8, help="Output channels")
    parser.add_argument("--xclbin", "-x", type=str, help="Path to xclbin file")
    parser.add_argument("--insts", "-i", type=str, help="Path to instructions file")
    
    args = parser.parse_args()
    
    use_aie = args.xclbin is not None and args.insts is not None
    
    success = test_aconv_layer(
        args.height,
        args.width,
        args.in_channels,
        args.out_channels,
        use_aie=use_aie,
        xclbin_path=args.xclbin,
        insts_path=args.insts,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

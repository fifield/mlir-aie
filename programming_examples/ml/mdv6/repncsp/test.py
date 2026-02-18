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
from mdv6.layers import RepNCSP


def bf16_to_uint16(bf16_tensor):
    """Convert bfloat16 tensor to uint16 array for AIE."""
    return bf16_tensor.view(torch.uint16).cpu().numpy()


def uint16_to_bf16(uint16_array):
    """Convert uint16 array from AIE to bfloat16 tensor."""
    return torch.from_numpy(uint16_array).view(torch.bfloat16)


def test_repncsp_layer(
    height,
    width,
    in_channels,
    out_channels,
    csp_expand=0.5,
    use_aie=False,
    xclbin_path=None,
    insts_path=None,
):
    """
    Test RepNCSP layer implementation.
    
    RepNCSP = Conv1 → Bottleneck → x1 + Conv2 → x2 → Concat → Conv3
    
    Args:
        height: Input height
        width: Input width
        in_channels: Number of input channels
        out_channels: Number of output channels
        csp_expand: CSP expansion factor (default 0.5)
        use_aie: Whether to run on AIE hardware
        xclbin_path: Path to xclbin file
        insts_path: Path to instructions file
    """
    
    neck_channels = int(out_channels * csp_expand)
    
    print(f"\n{'='*80}")
    print(f"Testing RepNCSP Layer:")
    print(f"  Input: ({height}, {width}, {in_channels})")
    print(f"  Output channels: {out_channels}")
    print(f"  CSP expand: {csp_expand} (neck={neck_channels})")
    print(f"  Architecture: Conv1 → Bottleneck → Concat → Conv3")
    print(f"{'='*80}\n")
    
    # Create PyTorch RepNCSP layer
    torch_repncsp = RepNCSP(in_channels, out_channels,
                            kernel_size=1,
                            csp_expand=csp_expand,
                            repeat_num=1)
    torch_repncsp.eval()
    
    # Convert model to bfloat16
    torch_repncsp = torch_repncsp.to(torch.bfloat16)
    
    # Generate random input (using bfloat16)
    torch.manual_seed(42)
    input_tensor = torch.randn(1, in_channels, height, width).to(torch.bfloat16)
    
    # Run PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_repncsp(input_tensor)
    
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
        input_hwc = input_tensor.squeeze(0).permute(1, 2, 0).contiguous()
        input_uint16 = bf16_to_uint16(input_hwc.flatten())
        
        # Extract weights - this is complex due to nested structure
        # Conv1 (1×1)
        conv1_weights = torch_repncsp.conv1.conv.weight.data
        conv1_bn_weight = torch_repncsp.conv1.bn.weight.data
        conv1_bn_bias = torch_repncsp.conv1.bn.bias.data
        conv1_bn_mean = torch_repncsp.conv1.bn.running_mean.data
        conv1_bn_var = torch_repncsp.conv1.bn.running_var.data
        
        # Bottleneck (nested in Sequential)
        bottleneck = torch_repncsp.bottleneck[0]  # First (and only) bottleneck
        
        # Bottleneck.conv1 = RepConv
        bn_conv3x3_weights = bottleneck.conv1.conv1.conv.weight.data
        bn_bn3x3_weight = bottleneck.conv1.conv1.bn.weight.data
        bn_bn3x3_bias = bottleneck.conv1.conv1.bn.bias.data
        bn_bn3x3_mean = bottleneck.conv1.conv1.bn.running_mean.data
        bn_bn3x3_var = bottleneck.conv1.conv1.bn.running_var.data
        
        bn_conv1x1_weights = bottleneck.conv1.conv2.conv.weight.data
        bn_bn1x1_weight = bottleneck.conv1.conv2.bn.weight.data
        bn_bn1x1_bias = bottleneck.conv1.conv2.bn.bias.data
        bn_bn1x1_mean = bottleneck.conv1.conv2.bn.running_mean.data
        bn_bn1x1_var = bottleneck.conv1.conv2.bn.running_var.data
        
        # Bottleneck.conv2
        bn_conv2_weights = bottleneck.conv2.conv.weight.data
        bn_bn2_weight = bottleneck.conv2.bn.weight.data
        bn_bn2_bias = bottleneck.conv2.bn.bias.data
        bn_bn2_mean = bottleneck.conv2.bn.running_mean.data
        bn_bn2_var = bottleneck.conv2.bn.running_var.data
        
        # Conv2 (1×1)
        conv2_weights = torch_repncsp.conv2.conv.weight.data
        conv2_bn_weight = torch_repncsp.conv2.bn.weight.data
        conv2_bn_bias = torch_repncsp.conv2.bn.bias.data
        conv2_bn_mean = torch_repncsp.conv2.bn.running_mean.data
        conv2_bn_var = torch_repncsp.conv2.bn.running_var.data
        
        # Conv3 (1×1)
        conv3_weights = torch_repncsp.conv3.conv.weight.data
        conv3_bn_weight = torch_repncsp.conv3.bn.weight.data
        conv3_bn_bias = torch_repncsp.conv3.bn.bias.data
        conv3_bn_mean = torch_repncsp.conv3.bn.running_mean.data
        conv3_bn_var = torch_repncsp.conv3.bn.running_var.data
        
        # Concatenate all weights in correct order
        weights_flat = torch.cat([
            conv1_weights.flatten(),
            conv1_bn_weight, conv1_bn_bias, conv1_bn_mean, conv1_bn_var,
            bn_conv3x3_weights.flatten(),
            bn_bn3x3_weight, bn_bn3x3_bias, bn_bn3x3_mean, bn_bn3x3_var,
            bn_conv1x1_weights.flatten(),
            bn_bn1x1_weight, bn_bn1x1_bias, bn_bn1x1_mean, bn_bn1x1_var,
            bn_conv2_weights.flatten(),
            bn_bn2_weight, bn_bn2_bias, bn_bn2_mean, bn_bn2_var,
            conv2_weights.flatten(),
            conv2_bn_weight, conv2_bn_bias, conv2_bn_mean, conv2_bn_var,
            conv3_weights.flatten(),
            conv3_bn_weight, conv3_bn_bias, conv3_bn_mean, conv3_bn_var,
        ])
        weights_uint16 = bf16_to_uint16(weights_flat)
        
        # Calculate buffer sizes
        input_size = height * width * in_channels
        output_size = height * width * out_channels
        
        print(f"Buffer sizes:")
        print(f"  Input:     {input_size} elements ({input_size * 2} bytes)")
        print(f"  Weights:   {len(weights_uint16)} elements ({len(weights_uint16) * 2} bytes)")
        print(f"  Output:    {output_size} elements ({output_size * 2} bytes)")
        print(f"  Neck size: {height * width * neck_channels} elements")
        
        # Setup AIE application
        print(f"\nSetting up AIE application...")
        print(f"  XCLBin: {xclbin_path}")
        print(f"  Instructions: {insts_path}")
        
        app = setup_aie(
            xclbin_path,
            insts_path,
            input_uint16.shape,
            np.uint16,
            weights_uint16.shape,
            np.uint16,
            (output_size,),
            np.uint16,
            kernel_name="MLIR_AIE"
        )
        
        # Execute on hardware
        print(f"\nExecuting kernel on NPU2...")
        start = time.time_ns()
        output_buffer = execute(app, input_uint16, weights_uint16)
        stop = time.time_ns()
        npu_time = (stop - start) / 1000
        
        print(f"  Execution time: {npu_time:.2f} μs ({npu_time/1000:.3f} ms)")
        
        # Convert output back to bfloat16
        output_bf16 = uint16_to_bf16(output_buffer[:output_size])
        
        # Reshape and convert to PyTorch format
        output_hwc = output_bf16.reshape(height, width, out_channels)
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
        
        # Higher tolerance due to many approximations
        tolerance = 0.40  # 4× sqrt + 3× sigmoid approximations
        if max_diff < tolerance:
            print(f"  ✓ PASS (max diff < {tolerance})")
            return True
        else:
            print(f"  ✗ FAIL (max diff >= {tolerance})")
            print(f"\nNote: Error is due to cumulative approximations:")
            print(f"  - 4× sqrt approximation (4 BatchNorms)")
            print(f"  - 3× sigmoid approximation (3 SiLU activations)")
            return False
    else:
        # CPU reference test
        print("\n✓ PyTorch reference test complete")
        return True


def main():
    parser = argparse.ArgumentParser(description="Test RepNCSP layer for MDV6")
    parser.add_argument("--height", "-ht", type=int, default=8, help="Input height")
    parser.add_argument("--width", "-wd", type=int, default=8, help="Input width")
    parser.add_argument("--in-channels", "-ic", type=int, default=16, help="Input channels")
    parser.add_argument("--out-channels", "-oc", type=int, default=16, help="Output channels")
    parser.add_argument("--csp-expand", "-e", type=float, default=0.5, help="CSP expansion factor")
    parser.add_argument("--xclbin", "-x", type=str, help="Path to xclbin file")
    parser.add_argument("--insts", "-i", type=str, help="Path to instructions file")
    
    args = parser.parse_args()
    
    use_aie = args.xclbin is not None and args.insts is not None
    
    success = test_repncsp_layer(
        args.height,
        args.width,
        args.in_channels,
        args.out_channels,
        args.csp_expand,
        use_aie=use_aie,
        xclbin_path=args.xclbin,
        insts_path=args.insts,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

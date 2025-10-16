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
from mdv6.layers import ELAN
from aie.utils.xrt import setup_aie, execute


def bf16_to_uint16(bf16_tensor):
    """Convert bfloat16 tensor to uint16 array for AIE."""
    return bf16_tensor.view(torch.uint16).cpu().numpy()


def uint16_to_bf16(uint16_array):
    """Convert uint16 array from AIE to bfloat16 tensor."""
    return torch.from_numpy(uint16_array).view(torch.bfloat16)


def test_elan_layer(
    height,
    width,
    in_channels,
    out_channels,
    part_channels,
    process_channels=None,
    use_aie=False,
    xclbin_path=None,
    insts_path=None,
):
    """
    Test ELAN layer implementation.
    
    ELAN = Conv1 → split → Conv2 → Conv3 → 4-way concat → Conv4
    """
    
    if process_channels is None:
        process_channels = part_channels // 2
    
    print(f"\n{'='*80}")
    print(f"Testing ELAN Layer:")
    print(f"  Input: ({height}, {width}, {in_channels})")
    print(f"  Output channels: {out_channels}")
    print(f"  Part channels: {part_channels}")
    print(f"  Process channels: {process_channels}")
    print(f"  Architecture: Conv1 → split → Conv2 → Conv3 → 4-way concat → Conv4")
    print(f"{'='*80}\n")
    
    # Create PyTorch ELAN layer
    torch_elan = ELAN(in_channels, out_channels, part_channels, process_channels)
    torch_elan.eval()
    torch_elan = torch_elan.to(torch.bfloat16)
    
    # Generate random input
    torch.manual_seed(42)
    input_tensor = torch.randn(1, in_channels, height, width).to(torch.bfloat16)
    
    # Run PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_elan(input_tensor)
    
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
        
        # Prepare data for AIE
        input_hwc = input_tensor.squeeze(0).permute(1, 2, 0).contiguous()
        input_uint16 = bf16_to_uint16(input_hwc.flatten())
        
        # Extract weights
        conv1_weights = torch_elan.conv1.conv.weight.data
        conv1_bn_weight = torch_elan.conv1.bn.weight.data
        conv1_bn_bias = torch_elan.conv1.bn.bias.data
        conv1_bn_mean = torch_elan.conv1.bn.running_mean.data
        conv1_bn_var = torch_elan.conv1.bn.running_var.data
        
        conv2_weights = torch_elan.conv2.conv.weight.data
        conv2_bn_weight = torch_elan.conv2.bn.weight.data
        conv2_bn_bias = torch_elan.conv2.bn.bias.data
        conv2_bn_mean = torch_elan.conv2.bn.running_mean.data
        conv2_bn_var = torch_elan.conv2.bn.running_var.data
        
        conv3_weights = torch_elan.conv3.conv.weight.data
        conv3_bn_weight = torch_elan.conv3.bn.weight.data
        conv3_bn_bias = torch_elan.conv3.bn.bias.data
        conv3_bn_mean = torch_elan.conv3.bn.running_mean.data
        conv3_bn_var = torch_elan.conv3.bn.running_var.data
        
        conv4_weights = torch_elan.conv4.conv.weight.data
        conv4_bn_weight = torch_elan.conv4.bn.weight.data
        conv4_bn_bias = torch_elan.conv4.bn.bias.data
        conv4_bn_mean = torch_elan.conv4.bn.running_mean.data
        conv4_bn_var = torch_elan.conv4.bn.running_var.data
        
        # Concatenate all weights
        weights_flat = torch.cat([
            conv1_weights.flatten(),
            conv1_bn_weight, conv1_bn_bias, conv1_bn_mean, conv1_bn_var,
            conv2_weights.flatten(),
            conv2_bn_weight, conv2_bn_bias, conv2_bn_mean, conv2_bn_var,
            conv3_weights.flatten(),
            conv3_bn_weight, conv3_bn_bias, conv3_bn_mean, conv3_bn_var,
            conv4_weights.flatten(),
            conv4_bn_weight, conv4_bn_bias, conv4_bn_mean, conv4_bn_var,
        ])
        weights_uint16 = bf16_to_uint16(weights_flat)
        
        # Calculate buffer sizes
        input_size = height * width * in_channels
        output_size = height * width * out_channels
        
        print(f"Buffer sizes:")
        print(f"  Input:   {input_size} elements ({input_size * 2} bytes)")
        print(f"  Weights: {len(weights_uint16)} elements ({len(weights_uint16) * 2} bytes)")
        print(f"  Output:  {output_size} elements ({output_size * 2} bytes)")
        
        # Setup AIE application
        print(f"\nSetting up AIE application...")
        app = setup_aie(
            xclbin_path, insts_path,
            input_uint16.shape, np.uint16,
            weights_uint16.shape, np.uint16,
            (output_size,), np.uint16,
            kernel_name="MLIR_AIE"
        )
        
        # Execute on hardware
        print(f"\nExecuting kernel on NPU2...")
        start = time.time_ns()
        output_buffer = execute(app, input_uint16, weights_uint16)
        stop = time.time_ns()
        npu_time = (stop - start) / 1000
        
        print(f"  Execution time: {npu_time:.2f} μs ({npu_time/1000:.3f} ms)")
        
        # Convert output back
        output_bf16 = uint16_to_bf16(output_buffer[:output_size])
        output_hwc = output_bf16.reshape(height, width, out_channels)
        aie_output = torch.from_numpy(output_hwc.float().numpy()).permute(2, 0, 1).unsqueeze(0)
        
        print(f"\nAIE output shape: {aie_output.shape}")
        print(f"AIE output range: [{aie_output.min():.4f}, {aie_output.max():.4f}]")
        print(f"AIE output mean: {aie_output.mean():.4f}, std: {aie_output.std():.4f}")
        
        # Compare
        diff = torch.abs(torch_output - aie_output)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"\nComparison (PyTorch vs AIE Hardware):")
        print(f"  Max absolute difference: {max_diff:.6f}")
        print(f"  Mean absolute difference: {mean_diff:.6f}")
        
        tolerance = 0.45  # 4× sqrt + 4× sigmoid + 4-way concat
        if max_diff < tolerance:
            print(f"  ✓ PASS (max diff < {tolerance})")
            return True
        else:
            print(f"  ✗ FAIL (max diff >= {tolerance})")
            print(f"\nNote: Error is due to cumulative approximations:")
            print(f"  - 4× sqrt approximation (4 BatchNorms)")
            print(f"  - 4× sigmoid approximation (4 SiLU activations)")
            print(f"  - 4-way concatenation with multiple feature scales")
            return False
    else:
        print("\n✓ PyTorch reference test complete")
        return True


def main():
    parser = argparse.ArgumentParser(description="Test ELAN layer for MDV6")
    parser.add_argument("--height", "-ht", type=int, default=8)
    parser.add_argument("--width", "-wd", type=int, default=8)
    parser.add_argument("--in-channels", "-ic", type=int, default=32)
    parser.add_argument("--out-channels", "-oc", type=int, default=32)
    parser.add_argument("--part-channels", "-pc", type=int, default=32)
    parser.add_argument("--process-channels", "-prc", type=int, default=None)
    parser.add_argument("--xclbin", "-x", type=str)
    parser.add_argument("--insts", "-i", type=str)
    
    args = parser.parse_args()
    
    use_aie = args.xclbin is not None and args.insts is not None
    
    success = test_elan_layer(
        args.height, args.width,
        args.in_channels, args.out_channels,
        args.part_channels, args.process_channels,
        use_aie=use_aie,
        xclbin_path=args.xclbin,
        insts_path=args.insts,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

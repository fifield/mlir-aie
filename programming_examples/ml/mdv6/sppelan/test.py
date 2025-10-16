#!/usr/bin/env python3
# sppelan/test.py
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.

import argparse
import numpy as np
import sys
import os

# Add parent directories to path to import mdv6
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../python"))

import torch
from mdv6.layers import SPPELAN
from aie.utils.xrt import setup_aie, execute


def bf16_to_uint16(tensor):
    """Convert bfloat16 tensor to uint16 numpy array."""
    return tensor.view(torch.int16).numpy().astype(np.uint16)


def uint16_to_bf16(array):
    """Convert uint16 numpy array to bfloat16 tensor."""
    return torch.from_numpy(array.astype(np.int16)).view(torch.bfloat16)


def test_sppelan(
    height=8,
    width=8,
    in_channels=16,
    out_channels=16,
    use_aie=False,
    print_output=False,
):
    """
    Test SPPELAN layer implementation.
    
    Args:
        height: Input height
        width: Input width
        in_channels: Input channels
        out_channels: Output channels
        use_aie: Whether to run on AIE hardware
        print_output: Whether to print output values
    """
    
    # neck_channels typically in_channels // 2
    neck_channels = in_channels // 2
    
    print(f"\n{'='*60}")
    print(f"Testing SPPELAN Layer")
    print(f"{'='*60}")
    print(f"Input shape: ({height}, {width}, {in_channels})")
    print(f"Output shape: ({height}, {width}, {out_channels})")
    print(f"Neck channels: {neck_channels}")
    print(f"MaxPool: kernel_size=5, stride=1, padding=2")
    
    # Create PyTorch SPPELAN layer
    layer = SPPELAN(in_channels, out_channels, neck_channels=neck_channels)
    layer.eval()
    layer = layer.to(torch.bfloat16)
    
    # Generate random input (NCHW format for PyTorch)
    torch.manual_seed(42)
    input_nchw = torch.randn(1, in_channels, height, width, dtype=torch.bfloat16)
    
    # Run PyTorch forward pass
    with torch.no_grad():
        output_nchw = layer(input_nchw)
    
    print(f"\nPyTorch output shape: {output_nchw.shape}")
    print(f"PyTorch output range: [{output_nchw.min():.4f}, {output_nchw.max():.4f}]")
    
    if use_aie:
        print(f"\n{'='*60}")
        print("Running on AIE Hardware")
        print(f"{'='*60}")
        
        # Convert to HWC format for AIE
        input_hwc = input_nchw.squeeze(0).permute(1, 2, 0).contiguous()
        output_hwc_ref = output_nchw.squeeze(0).permute(1, 2, 0).contiguous()
        
        # Extract weights from PyTorch model
        # Conv1 weights and BN parameters
        conv1_weight = layer.conv1.conv.weight.data  # (neck_channels, in_channels, 1, 1)
        conv1_bn_weight = layer.conv1.bn.weight.data
        conv1_bn_bias = layer.conv1.bn.bias.data
        conv1_bn_mean = layer.conv1.bn.running_mean.data
        conv1_bn_var = layer.conv1.bn.running_var.data
        
        # Conv5 weights and BN parameters
        conv5_weight = layer.conv5.conv.weight.data  # (out_channels, 4*neck_channels, 1, 1)
        conv5_bn_weight = layer.conv5.bn.weight.data
        conv5_bn_bias = layer.conv5.bn.bias.data
        conv5_bn_mean = layer.conv5.bn.running_mean.data
        conv5_bn_var = layer.conv5.bn.running_var.data
        
        # Reshape conv weights from (OC, IC, 1, 1) to (OC, IC)
        conv1_weight_2d = conv1_weight.squeeze(-1).squeeze(-1)  # (neck_channels, in_channels)
        conv5_weight_2d = conv5_weight.squeeze(-1).squeeze(-1)  # (out_channels, 4*neck_channels)
        
        # Concatenate all weights in order
        weights_list = [
            conv1_weight_2d.flatten(),
            conv1_bn_weight,
            conv1_bn_bias,
            conv1_bn_mean,
            conv1_bn_var,
            conv5_weight_2d.flatten(),
            conv5_bn_weight,
            conv5_bn_bias,
            conv5_bn_mean,
            conv5_bn_var,
        ]
        weights_concat = torch.cat(weights_list)
        
        print(f"\nWeight sizes:")
        print(f"  Conv1 weights: {conv1_weight_2d.numel()}")
        print(f"  Conv1 BN params: {4 * neck_channels}")
        print(f"  Conv5 weights: {conv5_weight_2d.numel()}")
        print(f"  Conv5 BN params: {4 * out_channels}")
        print(f"  Total weights: {weights_concat.numel()}")
        
        # Convert to uint16
        input_uint16 = bf16_to_uint16(input_hwc)
        weights_uint16 = bf16_to_uint16(weights_concat)
        
        # Calculate buffer sizes
        input_size = height * width * in_channels
        output_size = height * width * out_channels
        
        print(f"\nBuffer sizes:")
        print(f"  Input:   {input_size} elements ({input_size * 2} bytes)")
        print(f"  Weights: {len(weights_uint16)} elements ({len(weights_uint16) * 2} bytes)")
        print(f"  Output:  {output_size} elements ({output_size * 2} bytes)")
        
        # Setup AIE application
        print(f"\nSetting up AIE application...")
        app = setup_aie(
            "build/final.xclbin",
            "build/insts.bin",
            input_uint16.shape, np.uint16,
            weights_uint16.shape, np.uint16,
            (output_size,), np.uint16,
            kernel_name="MLIR_AIE"
        )
        
        # Execute on AIE
        print(f"\nExecuting kernel on NPU2...")
        import time
        start = time.time_ns()
        output_buffer = execute(app, input_uint16, weights_uint16)
        stop = time.time_ns()
        npu_time = (stop - start) / 1000
        
        print(f"  Execution time: {npu_time:.2f} μs ({npu_time/1000:.3f} ms)")
        
        # Extract output
        output_uint16 = output_buffer[:output_size]
        
        # Convert back to bfloat16
        output_hwc_aie = uint16_to_bf16(output_uint16).reshape(height, width, out_channels)
        
        print(f"\nAIE output shape: {output_hwc_aie.shape}")
        print(f"AIE output range: [{output_hwc_aie.min():.4f}, {output_hwc_aie.max():.4f}]")
        
        # Compare results
        abs_diff = torch.abs(output_hwc_ref - output_hwc_aie)
        
        max_abs_error = abs_diff.max().item()
        mean_abs_error = abs_diff.mean().item()
        
        # Calculate relative error only for non-zero reference values
        ref_abs = torch.abs(output_hwc_ref)
        mask = ref_abs > 0.01  # Only calculate relative error for values > 0.01
        if mask.any():
            rel_diff = abs_diff[mask] / ref_abs[mask]
            max_rel_error = rel_diff.max().item() * 100
            mean_rel_error = rel_diff.mean().item() * 100
        else:
            max_rel_error = 0.0
            mean_rel_error = 0.0
        
        print(f"\n{'='*60}")
        print("Accuracy Comparison")
        print(f"{'='*60}")
        print(f"Max absolute error: {max_abs_error:.6f}")
        print(f"Mean absolute error: {mean_abs_error:.6f}")
        print(f"Max relative error: {max_rel_error:.2f}% (for values > 0.01)")
        print(f"Mean relative error: {mean_rel_error:.2f}% (for values > 0.01)")
        
        if print_output:
            print(f"\n{'='*60}")
            print("Sample Output Values (first 5 elements)")
            print(f"{'='*60}")
            ref_flat = output_hwc_ref.flatten()[:5]
            aie_flat = output_hwc_aie.flatten()[:5]
            for i in range(5):
                print(f"  [{i}] PyTorch: {ref_flat[i]:.6f}, AIE: {aie_flat[i]:.6f}, "
                      f"Diff: {abs(ref_flat[i] - aie_flat[i]):.6f}")
        
        # Check tolerance using absolute error (more reliable for SPPELAN)
        # SPPELAN has 2× sqrt + 2× sigmoid approximations
        abs_tolerance = 0.15  # 0.15 absolute error tolerance
        if max_abs_error < abs_tolerance:
            print(f"\n✓ Test PASSED (max absolute error {max_abs_error:.6f} < {abs_tolerance})")
            return True
        else:
            print(f"\n✗ Test FAILED (max absolute error {max_abs_error:.6f} >= {abs_tolerance})")
            print("  Note: SPPELAN uses fast sqrt and sigmoid approximations")
            return False
    else:
        print("\nSkipping AIE execution (use --use-aie to run on hardware)")
        return True


def main():
    parser = argparse.ArgumentParser(description="Test SPPELAN layer")
    parser.add_argument("--height", type=int, default=8, help="Input height")
    parser.add_argument("--width", type=int, default=8, help="Input width")
    parser.add_argument("--in-channels", type=int, default=16, help="Input channels")
    parser.add_argument("--out-channels", type=int, default=16, help="Output channels")
    parser.add_argument("--use-aie", action="store_true", help="Run on AIE hardware")
    parser.add_argument("--print-output", action="store_true", help="Print output values")
    
    args = parser.parse_args()
    
    success = test_sppelan(
        height=args.height,
        width=args.width,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        use_aie=args.use_aie,
        print_output=args.print_output,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

"""
Test Conv layer with MDV6 model dimensions.

This script specifically tests the Conv0 and Conv1 configurations used in the
actual MDV6 model to ensure they work correctly.

Conv0: 8×8, 3→32 (first conv layer - RGB input)
Conv1: 8×8, 32→64 (second conv layer)
"""

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
import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime


def bf16_to_uint16(bf16_tensor):
    """Convert bfloat16 tensor to uint16 array for AIE."""
    return bf16_tensor.view(torch.uint16).cpu().numpy()


def uint16_to_bf16(uint16_array):
    """Convert uint16 array from AIE to bfloat16 tensor."""
    return torch.from_numpy(uint16_array.copy()).view(torch.bfloat16)


def test_conv_configuration(
    config_name,
    input_height,
    input_width,
    input_channels,
    output_channels,
    kernel_size=3,
    stride=1,
    padding=1,
    use_aie=False,
    xclbin_path=None,
    insts_path=None,
    verbose=False,
):
    """
    Test a specific Conv configuration.
    
    Args:
        config_name: Name of the configuration (e.g., "Conv0", "Conv1")
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
        verbose: Print detailed output
    
    Returns:
        bool: True if test passed, False otherwise
    """
    
    print(f"\n{'='*80}")
    print(f"Testing {config_name} Configuration:")
    print(f"  Input: ({input_height}, {input_width}, {input_channels})")
    print(f"  Output channels: {output_channels}")
    print(f"  Kernel size: {kernel_size}x{kernel_size}")
    print(f"  Stride: {stride}, Padding: {padding}")
    print(f"{'='*80}\n")
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    
    # Calculate memory requirements
    input_size = input_height * input_width * input_channels
    weight_size = output_channels * input_channels * kernel_size * kernel_size
    output_size = output_height * output_width * output_channels
    
    input_bytes = input_size * 2  # bf16 = 2 bytes
    weight_bytes = weight_size * 2
    output_bytes = output_size * 2
    total_bytes = input_bytes + weight_bytes + output_bytes
    
    print(f"Memory Requirements:")
    print(f"  Input:   {input_size:6d} elements ({input_bytes:8d} bytes = {input_bytes/1024:6.2f} KB)")
    print(f"  Weights: {weight_size:6d} elements ({weight_bytes:8d} bytes = {weight_bytes/1024:6.2f} KB)")
    print(f"  Output:  {output_size:6d} elements ({output_bytes:8d} bytes = {output_bytes/1024:6.2f} KB)")
    print(f"  Total:   {total_bytes:8d} bytes = {total_bytes/1024:6.2f} KB")
    
    # Check if it fits in L1 (64 KB)
    L1_SIZE = 64 * 1024
    if total_bytes <= L1_SIZE:
        print(f"  ✓ Fits in L1 memory (64 KB)")
    else:
        print(f"  ✗ Exceeds L1 memory (64 KB) - requires tiling")
    print()
    
    # Create PyTorch Conv layer
    torch_conv = Conv(
        input_channels,
        output_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        activation="SiLU"
    )
    torch_conv.eval()
    
    # Convert model to bfloat16
    torch_conv = torch_conv.to(torch.bfloat16)
    
    # Generate random input (using bfloat16)
    torch.manual_seed(42)
    input_tensor = torch.randn(1, input_channels, input_height, input_width).to(torch.bfloat16)
    
    # Run PyTorch forward pass
    with torch.no_grad():
        # Get conv output before activation (for comparison with AIE kernel)
        conv_out = torch_conv.bn(torch_conv.conv(input_tensor))
        torch_output = conv_out
    
    if verbose:
        print(f"PyTorch output shape: {torch_output.shape}")
        print(f"PyTorch output range: [{torch_output.min():.4f}, {torch_output.max():.4f}]")
        print(f"PyTorch output mean: {torch_output.mean():.4f}, std: {torch_output.std():.4f}")
    
    if use_aie:
        if xclbin_path is None or insts_path is None:
            print("ERROR: xclbin_path and insts_path required for AIE execution")
            return False
        
        print(f"Running on NPU2 Hardware (AIE2P)")
        print(f"  XCLBin: {xclbin_path}")
        print(f"  Instructions: {insts_path}\n")
        
        # Prepare data for AIE (convert to HWC format and uint16)
        input_hwc = input_tensor.squeeze(0).permute(1, 2, 0).contiguous()
        input_uint16 = bf16_to_uint16(input_hwc.flatten())
        
        # Extract weights
        weights = torch_conv.conv.weight.data
        weights_uint16 = bf16_to_uint16(weights.flatten())
        
        # Setup AIE application
        print(f"Setting up AIE application...")
        npu_kernel = NPUKernel(xclbin_path, insts_path, kernel_name="MLIR_AIE")
        kernel_handle = DefaultNPURuntime.load(npu_kernel)

        # Execute on hardware
        print(f"Executing kernel on NPU2...")
        in1 = iron.tensor(input_uint16, dtype=np.uint16)
        in2 = iron.tensor(weights_uint16, dtype=np.uint16)
        out = iron.zeros(output_size, dtype=np.uint16)
        start = time.time_ns()
        ret = DefaultNPURuntime.run(kernel_handle, [in1, in2, out])
        stop = time.time_ns()
        npu_time = (stop - start) / 1000

        print(f"  Execution time: {npu_time:.2f} μs ({npu_time/1000:.3f} ms)")
        output_buffer = out.numpy()
        
        # Convert output back to bfloat16
        output_bf16 = uint16_to_bf16(output_buffer[:output_size])
        output_hwc = output_bf16.reshape(output_height, output_width, output_channels)
        aie_output = torch.from_numpy(output_hwc.float().numpy()).permute(2, 0, 1).unsqueeze(0)
        
        if verbose:
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
        tolerance = 0.1
        if max_diff < tolerance:
            print(f"  ✓ PASS (max diff < {tolerance})")
            return True
        else:
            print(f"  ✗ FAIL (max diff >= {tolerance})")
            return False
    else:
        # CPU reference test
        print("Running CPU reference (NumPy)...")
        
        # Simple reference implementation
        input_np = input_tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        weights_np = torch_conv.conv.weight.data.float().cpu().numpy()
        
        output_np = np.zeros((output_height, output_width, output_channels), dtype=np.float32)
        
        # Simple convolution
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
        
        # Apply BatchNorm
        bn_weight = torch_conv.bn.weight.data.float().cpu().numpy()
        bn_bias = torch_conv.bn.bias.data.float().cpu().numpy()
        for oc in range(output_channels):
            output_np[:, :, oc] = output_np[:, :, oc] * bn_weight[oc] + bn_bias[oc]
        
        # Convert back to tensor
        output_np_tensor = torch.from_numpy(output_np).permute(2, 0, 1).unsqueeze(0)
        
        if verbose:
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
        
        tolerance = 0.1
        if max_diff < tolerance:
            print(f"  ✓ PASS (max diff < {tolerance})")
            return True
        else:
            print(f"  ✗ FAIL (max diff >= {tolerance})")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Conv layer with MDV6 model dimensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test Conv0 (CPU reference)
  python3 test_model_dimensions.py --config conv0
  
  # Test Conv1 (CPU reference)
  python3 test_model_dimensions.py --config conv1
  
  # Test both configurations
  python3 test_model_dimensions.py --config all
  
  # Test Conv0 on AIE hardware
  python3 test_model_dimensions.py --config conv0 \\
      --xclbin build/conv_3_32.xclbin \\
      --insts build/conv_3_32.insts.bin
  
  # Test Conv1 on AIE hardware
  python3 test_model_dimensions.py --config conv1 \\
      --xclbin build/conv_32_64.xclbin \\
      --insts build/conv_32_64.insts.bin
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        choices=["conv0", "conv1", "all"],
        default="all",
        help="Which configuration to test (default: all)"
    )
    parser.add_argument("--xclbin", "-x", type=str, help="Path to xclbin file")
    parser.add_argument("--insts", "-i", type=str, help="Path to instructions file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    use_aie = args.xclbin is not None and args.insts is not None
    
    # Define MDV6 model configurations
    configs = {
        "conv0": {
            "name": "Conv0",
            "height": 8,
            "width": 8,
            "in_channels": 3,
            "out_channels": 32,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        },
        "conv1": {
            "name": "Conv1",
            "height": 8,
            "width": 8,
            "in_channels": 32,
            "out_channels": 64,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        },
    }
    
    # Determine which configs to test
    if args.config == "all":
        configs_to_test = ["conv0", "conv1"]
    else:
        configs_to_test = [args.config]
    
    # Run tests
    results = {}
    for config_key in configs_to_test:
        config = configs[config_key]
        
        success = test_conv_configuration(
            config["name"],
            config["height"],
            config["width"],
            config["in_channels"],
            config["out_channels"],
            config["kernel_size"],
            config["stride"],
            config["padding"],
            use_aie=use_aie,
            xclbin_path=args.xclbin,
            insts_path=args.insts,
            verbose=args.verbose,
        )
        
        results[config["name"]] = success
    
    # Print summary
    print(f"\n{'='*80}")
    print("Test Summary:")
    print(f"{'='*80}")
    for config_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {config_name}: {status}")
    print(f"{'='*80}\n")
    
    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

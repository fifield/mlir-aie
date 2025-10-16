#!/usr/bin/env python3
# repncsp_elan/test.py
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
from mdv6.layers import RepNCSPELAN
from aie.utils.xrt import setup_aie, execute


def bf16_to_uint16(tensor):
    """Convert bfloat16 tensor to uint16 numpy array."""
    return tensor.view(torch.uint16).cpu().numpy()


def uint16_to_bf16(array):
    """Convert uint16 numpy array to bfloat16 tensor."""
    return torch.from_numpy(array).view(torch.bfloat16)


def extract_repncsp_weights(repncsp_module):
    """
    Extract weights from a RepNCSP module.
    
    RepNCSP structure:
        - conv1 (1×1) + BN
        - bottleneck (RepConv + Conv + residual)
        - conv2 (1×1) + BN (bypass)
        - conv3 (1×1) + BN (merge)
    """
    weights_list = []
    
    # Conv1 weights and BN
    conv1_w = repncsp_module.conv1.conv.weight.data.squeeze(-1).squeeze(-1).flatten()
    conv1_bn_w = repncsp_module.conv1.bn.weight.data
    conv1_bn_b = repncsp_module.conv1.bn.bias.data
    conv1_bn_m = repncsp_module.conv1.bn.running_mean.data
    conv1_bn_v = repncsp_module.conv1.bn.running_var.data
    weights_list.extend([conv1_w, conv1_bn_w, conv1_bn_b, conv1_bn_m, conv1_bn_v])
    
    # Bottleneck weights (RepConv + Conv2)
    bottleneck = repncsp_module.bottleneck[0]  # First (and only) bottleneck
    
    # RepConv: conv1 (3×3) + BN
    repconv_conv1_w = bottleneck.conv1.conv1.conv.weight.data.flatten()
    repconv_bn1_w = bottleneck.conv1.conv1.bn.weight.data
    repconv_bn1_b = bottleneck.conv1.conv1.bn.bias.data
    repconv_bn1_m = bottleneck.conv1.conv1.bn.running_mean.data
    repconv_bn1_v = bottleneck.conv1.conv1.bn.running_var.data
    weights_list.extend([repconv_conv1_w, repconv_bn1_w, repconv_bn1_b, repconv_bn1_m, repconv_bn1_v])
    
    # RepConv: conv2 (1×1) + BN
    repconv_conv2_w = bottleneck.conv1.conv2.conv.weight.data.squeeze(-1).squeeze(-1).flatten()
    repconv_bn2_w = bottleneck.conv1.conv2.bn.weight.data
    repconv_bn2_b = bottleneck.conv1.conv2.bn.bias.data
    repconv_bn2_m = bottleneck.conv1.conv2.bn.running_mean.data
    repconv_bn2_v = bottleneck.conv1.conv2.bn.running_var.data
    weights_list.extend([repconv_conv2_w, repconv_bn2_w, repconv_bn2_b, repconv_bn2_m, repconv_bn2_v])
    
    # Bottleneck: conv2 (3×3) + BN
    bn_conv2_w = bottleneck.conv2.conv.weight.data.flatten()
    bn_conv2_bn_w = bottleneck.conv2.bn.weight.data
    bn_conv2_bn_b = bottleneck.conv2.bn.bias.data
    bn_conv2_bn_m = bottleneck.conv2.bn.running_mean.data
    bn_conv2_bn_v = bottleneck.conv2.bn.running_var.data
    weights_list.extend([bn_conv2_w, bn_conv2_bn_w, bn_conv2_bn_b, bn_conv2_bn_m, bn_conv2_bn_v])
    
    # Conv2 (1×1) + BN (bypass path)
    conv2_w = repncsp_module.conv2.conv.weight.data.squeeze(-1).squeeze(-1).flatten()
    conv2_bn_w = repncsp_module.conv2.bn.weight.data
    conv2_bn_b = repncsp_module.conv2.bn.bias.data
    conv2_bn_m = repncsp_module.conv2.bn.running_mean.data
    conv2_bn_v = repncsp_module.conv2.bn.running_var.data
    weights_list.extend([conv2_w, conv2_bn_w, conv2_bn_b, conv2_bn_m, conv2_bn_v])
    
    # Conv3 (1×1) + BN (merge)
    conv3_w = repncsp_module.conv3.conv.weight.data.squeeze(-1).squeeze(-1).flatten()
    conv3_bn_w = repncsp_module.conv3.bn.weight.data
    conv3_bn_b = repncsp_module.conv3.bn.bias.data
    conv3_bn_m = repncsp_module.conv3.bn.running_mean.data
    conv3_bn_v = repncsp_module.conv3.bn.running_var.data
    weights_list.extend([conv3_w, conv3_bn_w, conv3_bn_b, conv3_bn_m, conv3_bn_v])
    
    return torch.cat(weights_list)


def test_repncsp_elan(
    height=8,
    width=8,
    in_channels=32,
    out_channels=32,
    part_channels=32,
    process_channels=None,
    use_aie=False,
    print_output=False,
):
    """
    Test RepNCSPELAN layer implementation.
    
    Args:
        height: Input height
        width: Input width
        in_channels: Input channels
        out_channels: Output channels
        part_channels: Intermediate channels after Conv1
        process_channels: Processing channels (default: part_channels // 2)
        use_aie: Whether to run on AIE hardware
        print_output: Whether to print output values
    """
    
    if process_channels is None:
        process_channels = part_channels // 2
    
    print(f"\n{'='*60}")
    print(f"Testing RepNCSPELAN Layer")
    print(f"{'='*60}")
    print(f"Input shape: ({height}, {width}, {in_channels})")
    print(f"Output shape: ({height}, {width}, {out_channels})")
    print(f"Part channels: {part_channels}")
    print(f"Process channels: {process_channels}")
    print(f"Architecture: Conv1 → split → RepNCSP → Conv3x3 → RepNCSP → Conv3x3 → 4-way concat → Conv4")
    
    # Create PyTorch RepNCSPELAN layer
    layer = RepNCSPELAN(in_channels, out_channels, part_channels, process_channels)
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
        weights_list = []
        
        # Conv1 weights and BN
        conv1_w = layer.conv1.conv.weight.data.squeeze(-1).squeeze(-1).flatten()
        conv1_bn_w = layer.conv1.bn.weight.data
        conv1_bn_b = layer.conv1.bn.bias.data
        conv1_bn_m = layer.conv1.bn.running_mean.data
        conv1_bn_v = layer.conv1.bn.running_var.data
        weights_list.extend([conv1_w, conv1_bn_w, conv1_bn_b, conv1_bn_m, conv1_bn_v])
        
        # RepNCSP #1 (conv2[0]) + Conv3x3 (conv2[1])
        repncsp1 = layer.conv2[0]
        repncsp1_weights = extract_repncsp_weights(repncsp1)
        weights_list.append(repncsp1_weights)
        
        conv3x3_1_w = layer.conv2[1].conv.weight.data.flatten()
        conv3x3_1_bn_w = layer.conv2[1].bn.weight.data
        conv3x3_1_bn_b = layer.conv2[1].bn.bias.data
        conv3x3_1_bn_m = layer.conv2[1].bn.running_mean.data
        conv3x3_1_bn_v = layer.conv2[1].bn.running_var.data
        weights_list.extend([conv3x3_1_w, conv3x3_1_bn_w, conv3x3_1_bn_b, conv3x3_1_bn_m, conv3x3_1_bn_v])
        
        # RepNCSP #2 (conv3[0]) + Conv3x3 (conv3[1])
        repncsp2 = layer.conv3[0]
        repncsp2_weights = extract_repncsp_weights(repncsp2)
        weights_list.append(repncsp2_weights)
        
        conv3x3_2_w = layer.conv3[1].conv.weight.data.flatten()
        conv3x3_2_bn_w = layer.conv3[1].bn.weight.data
        conv3x3_2_bn_b = layer.conv3[1].bn.bias.data
        conv3x3_2_bn_m = layer.conv3[1].bn.running_mean.data
        conv3x3_2_bn_v = layer.conv3[1].bn.running_var.data
        weights_list.extend([conv3x3_2_w, conv3x3_2_bn_w, conv3x3_2_bn_b, conv3x3_2_bn_m, conv3x3_2_bn_v])
        
        # Conv4 weights and BN
        conv4_w = layer.conv4.conv.weight.data.squeeze(-1).squeeze(-1).flatten()
        conv4_bn_w = layer.conv4.bn.weight.data
        conv4_bn_b = layer.conv4.bn.bias.data
        conv4_bn_m = layer.conv4.bn.running_mean.data
        conv4_bn_v = layer.conv4.bn.running_var.data
        weights_list.extend([conv4_w, conv4_bn_w, conv4_bn_b, conv4_bn_m, conv4_bn_v])
        
        # Concatenate all weights
        weights_concat = torch.cat(weights_list)
        
        print(f"\nWeight sizes:")
        print(f"  Conv1: {conv1_w.numel()} weights + {4 * part_channels} BN params")
        print(f"  RepNCSP #1: {repncsp1_weights.numel()} params")
        print(f"  Conv3x3 #1: {conv3x3_1_w.numel()} weights + {4 * process_channels} BN params")
        print(f"  RepNCSP #2: {repncsp2_weights.numel()} params")
        print(f"  Conv3x3 #2: {conv3x3_2_w.numel()} weights + {4 * process_channels} BN params")
        print(f"  Conv4: {conv4_w.numel()} weights + {4 * out_channels} BN params")
        print(f"  Total weights: {weights_concat.numel()}")
        
        # Convert to uint16
        input_uint16 = bf16_to_uint16(input_hwc.flatten())
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
        mask = ref_abs > 0.01
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
        
        # Check tolerance using absolute error
        # RepNCSPELAN has many approximations: ~12× sqrt, ~10× sigmoid
        abs_tolerance = 0.30  # 0.3 absolute error tolerance
        if max_abs_error < abs_tolerance:
            print(f"\n✓ Test PASSED (max absolute error {max_abs_error:.6f} < {abs_tolerance})")
            return True
        else:
            print(f"\n✗ Test FAILED (max absolute error {max_abs_error:.6f} >= {abs_tolerance})")
            print("  Note: RepNCSPELAN has many approximations (~12× sqrt, ~10× sigmoid)")
            return False
    else:
        print("\n✓ PyTorch reference test complete")
        return True


def main():
    parser = argparse.ArgumentParser(description="Test RepNCSPELAN layer")
    parser.add_argument("--height", type=int, default=8, help="Input height")
    parser.add_argument("--width", type=int, default=8, help="Input width")
    parser.add_argument("--in-channels", type=int, default=32, help="Input channels")
    parser.add_argument("--out-channels", type=int, default=32, help="Output channels")
    parser.add_argument("--part-channels", type=int, default=32, help="Part channels")
    parser.add_argument("--process-channels", type=int, default=None, help="Process channels")
    parser.add_argument("--use-aie", action="store_true", help="Run on AIE hardware")
    parser.add_argument("--print-output", action="store_true", help="Print output values")
    
    args = parser.parse_args()
    
    success = test_repncsp_elan(
        height=args.height,
        width=args.width,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        part_channels=args.part_channels,
        process_channels=args.process_channels,
        use_aie=args.use_aie,
        print_output=args.print_output,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

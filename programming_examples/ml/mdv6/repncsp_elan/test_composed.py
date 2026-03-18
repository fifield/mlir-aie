#!/usr/bin/env python3
"""
Test RepNCSPELAN using host-orchestrated composition of simpler layers.

Instead of one monolithic kernel, we compose RepNCSPELAN from:
  Conv1x1 → split → RepNCSP → Conv3x3 → RepNCSP → Conv3x3 → concat → Conv1x1

Each sub-layer reuses the existing conv/repncsp kernels.
This approach stays within the 64KB tile memory limit by loading one
sub-layer's weights at a time.
"""

import argparse
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../python"))

import torch
from mdv6.layers import RepNCSPELAN

import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime


def bf16_to_uint16(tensor):
    return tensor.view(torch.uint16).cpu().numpy()


def uint16_to_bf16(array):
    return torch.from_numpy(array.copy()).view(torch.bfloat16)


def run_conv1x1_on_npu(kernel_handle, input_hwc, weights_and_bn, height, width, in_ch, out_ch):
    """Run a Conv1x1+BN+SiLU sub-layer on NPU using the conv kernel."""
    input_uint16 = bf16_to_uint16(input_hwc.flatten())
    weights_uint16 = bf16_to_uint16(weights_and_bn)
    output_size = height * width * out_ch

    in1 = iron.tensor(input_uint16, dtype=np.uint16)
    in2 = iron.tensor(weights_uint16, dtype=np.uint16)
    out = iron.zeros(output_size, dtype=np.uint16)

    ret = DefaultNPURuntime.run(kernel_handle, [in1, in2, out])
    output_data = out.numpy()[:output_size].copy()
    return uint16_to_bf16(output_data).reshape(height, width, out_ch)


def extract_conv_weights_fused(conv_module):
    """Extract Conv + fused BN weights as [conv_wts, bn_w_fused, bn_b_fused]."""
    conv_w = conv_module.conv.weight.data.flatten()
    # Fuse BN: bn_w_fused = gamma / sqrt(var + eps), bn_b_fused = beta - gamma * mean / sqrt(var + eps)
    eps = conv_module.bn.eps
    gamma = conv_module.bn.weight.data
    beta = conv_module.bn.bias.data
    mean = conv_module.bn.running_mean.data
    var = conv_module.bn.running_var.data
    inv_std = 1.0 / torch.sqrt(var + eps)
    bn_w_fused = gamma * inv_std
    bn_b_fused = beta - gamma * mean * inv_std
    return bf16_to_uint16(torch.cat([conv_w, bn_w_fused, bn_b_fused]))


def extract_repncsp_weights(repncsp):
    """Extract all RepNCSP weights as packed bf16 uint16 array."""
    parts = []
    # Conv1
    parts.append(repncsp.conv1.conv.weight.data.flatten())
    parts.extend([repncsp.conv1.bn.weight.data, repncsp.conv1.bn.bias.data,
                  repncsp.conv1.bn.running_mean.data, repncsp.conv1.bn.running_var.data])
    # Bottleneck[0].conv1 = RepConv (conv3x3+BN, conv1x1+BN)
    bn = repncsp.bottleneck[0]
    parts.append(bn.conv1.conv1.conv.weight.data.flatten())
    parts.extend([bn.conv1.conv1.bn.weight.data, bn.conv1.conv1.bn.bias.data,
                  bn.conv1.conv1.bn.running_mean.data, bn.conv1.conv1.bn.running_var.data])
    parts.append(bn.conv1.conv2.conv.weight.data.flatten())
    parts.extend([bn.conv1.conv2.bn.weight.data, bn.conv1.conv2.bn.bias.data,
                  bn.conv1.conv2.bn.running_mean.data, bn.conv1.conv2.bn.running_var.data])
    # Bottleneck[0].conv2
    parts.append(bn.conv2.conv.weight.data.flatten())
    parts.extend([bn.conv2.bn.weight.data, bn.conv2.bn.bias.data,
                  bn.conv2.bn.running_mean.data, bn.conv2.bn.running_var.data])
    # Conv2
    parts.append(repncsp.conv2.conv.weight.data.flatten())
    parts.extend([repncsp.conv2.bn.weight.data, repncsp.conv2.bn.bias.data,
                  repncsp.conv2.bn.running_mean.data, repncsp.conv2.bn.running_var.data])
    # Conv3
    parts.append(repncsp.conv3.conv.weight.data.flatten())
    parts.extend([repncsp.conv3.bn.weight.data, repncsp.conv3.bn.bias.data,
                  repncsp.conv3.bn.running_mean.data, repncsp.conv3.bn.running_var.data])
    return bf16_to_uint16(torch.cat(parts))


def run_sublayer(kernel_handle, input_uint16, weights_uint16, output_size):
    """Run a single sub-layer on NPU."""
    in1 = iron.tensor(input_uint16, dtype=np.uint16)
    in2 = iron.tensor(weights_uint16, dtype=np.uint16)
    out = iron.zeros(output_size, dtype=np.uint16)
    DefaultNPURuntime.run(kernel_handle, [in1, in2, out])
    return out.numpy()[:output_size].copy()


def test_repncsp_elan_composed(
    height=8, width=8,
    in_channels=32, out_channels=32,
    part_channels=32, process_channels=None,
    use_aie=False,
):
    if process_channels is None:
        process_channels = part_channels // 2
    half_part = part_channels // 2
    concat_channels = part_channels + 2 * process_channels

    print(f"\n{'='*60}")
    print(f"Testing RepNCSPELAN ({'NPU Composed' if use_aie else 'CPU Reference'})")
    print(f"{'='*60}")
    print(f"Input: ({height}, {width}, {in_channels}) → Output: ({height}, {width}, {out_channels})")
    print(f"Part: {part_channels}, Process: {process_channels}")

    # PyTorch reference
    layer = RepNCSPELAN(in_channels, out_channels, part_channels, process_channels)
    layer.eval()
    layer = layer.to(torch.bfloat16)

    torch.manual_seed(42)
    input_nchw = torch.randn(1, in_channels, height, width, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_output = layer(input_nchw)

    print(f"\nPyTorch ref range: [{ref_output.min():.4f}, {ref_output.max():.4f}]")

    if not use_aie:
        print("\n✓ CPU reference test complete")
        return True

    # ================================================================
    # Host-orchestrated NPU execution using sub-layer xclbins
    # ================================================================
    print(f"\nRunning 6 sub-layers on NPU...")

    base = os.path.dirname(__file__)
    conv_dir = os.path.join(base, "..", "conv", "build")
    repncsp_dir = os.path.join(base, "..", "repncsp", "build")

    # Load fused Conv+BN+SiLU kernel handles
    kh_conv1 = DefaultNPURuntime.load(NPUKernel(
        f"{conv_dir}/fused_1x1_32_32.xclbin", f"{conv_dir}/fused_1x1_32_32.bin"))
    kh_repncsp = DefaultNPURuntime.load(NPUKernel(
        f"{repncsp_dir}/repncsp_16_16.xclbin", f"{repncsp_dir}/repncsp_16_16.bin"))
    kh_conv3x3 = DefaultNPURuntime.load(NPUKernel(
        f"{conv_dir}/fused_3x3_16_16.xclbin", f"{conv_dir}/fused_3x3_16_16.bin"))
    kh_conv4 = DefaultNPURuntime.load(NPUKernel(
        f"{conv_dir}/fused_1x1_64_32.xclbin", f"{conv_dir}/fused_1x1_64_32.bin"))

    # Convert input to HWC uint16
    input_hwc = input_nchw.squeeze(0).permute(1, 2, 0).contiguous()
    input_uint16 = bf16_to_uint16(input_hwc.flatten())

    total_time = 0

    # Stage 1: Conv1 (1x1, in→part)
    print("  1. Conv1 (1x1)...", end=" ", flush=True)
    t0 = time.time_ns()
    conv1_out_u16 = run_sublayer(kh_conv1, input_uint16,
                                  extract_conv_weights_fused(layer.conv1),
                                  height * width * part_channels)
    dt = time.time_ns() - t0; total_time += dt
    print(f"{dt/1000:.0f} μs")

    # Split conv1 output into x1, x2 (channel split in HWC)
    conv1_bf16 = uint16_to_bf16(conv1_out_u16).reshape(height, width, part_channels)
    x1_bf16 = conv1_bf16[:, :, :half_part].contiguous()
    x2_bf16 = conv1_bf16[:, :, half_part:].contiguous()
    x2_u16 = bf16_to_uint16(x2_bf16.flatten())

    # Stage 2: RepNCSP#1 (half_part→process)
    print("  2. RepNCSP#1...", end=" ", flush=True)
    t0 = time.time_ns()
    x3_rn_u16 = run_sublayer(kh_repncsp, x2_u16,
                              extract_repncsp_weights(layer.conv2[0]),
                              height * width * process_channels)
    dt = time.time_ns() - t0; total_time += dt
    print(f"{dt/1000:.0f} μs")

    # Stage 3: Conv3x3#1 (process→process)
    print("  3. Conv3x3#1...", end=" ", flush=True)
    t0 = time.time_ns()
    x3_u16 = run_sublayer(kh_conv3x3, x3_rn_u16,
                           extract_conv_weights_fused(layer.conv2[1]),
                           height * width * process_channels)
    dt = time.time_ns() - t0; total_time += dt
    print(f"{dt/1000:.0f} μs")

    # Stage 4: RepNCSP#2 (process→process)
    print("  4. RepNCSP#2...", end=" ", flush=True)
    t0 = time.time_ns()
    x4_rn_u16 = run_sublayer(kh_repncsp, x3_u16,
                              extract_repncsp_weights(layer.conv3[0]),
                              height * width * process_channels)
    dt = time.time_ns() - t0; total_time += dt
    print(f"{dt/1000:.0f} μs")

    # Stage 5: Conv3x3#2 (process→process)
    print("  5. Conv3x3#2...", end=" ", flush=True)
    t0 = time.time_ns()
    x4_u16 = run_sublayer(kh_conv3x3, x4_rn_u16,
                           extract_conv_weights_fused(layer.conv3[1]),
                           height * width * process_channels)
    dt = time.time_ns() - t0; total_time += dt
    print(f"{dt/1000:.0f} μs")

    # Stage 6: Concat [x1,x2,x3,x4] → Conv4 (1x1, concat→out)
    x3_bf16 = uint16_to_bf16(x3_u16).reshape(height, width, process_channels)
    x4_bf16 = uint16_to_bf16(x4_u16).reshape(height, width, process_channels)
    concat_bf16 = torch.cat([x1_bf16, x2_bf16, x3_bf16, x4_bf16], dim=2)
    concat_u16 = bf16_to_uint16(concat_bf16.flatten())

    print("  6. Conv4 (1x1)...", end=" ", flush=True)
    t0 = time.time_ns()
    final_u16 = run_sublayer(kh_conv4, concat_u16,
                              extract_conv_weights_fused(layer.conv4),
                              height * width * out_channels)
    dt = time.time_ns() - t0; total_time += dt
    print(f"{dt/1000:.0f} μs")

    print(f"\n  Total: {total_time/1000:.0f} μs ({total_time/1e6:.1f} ms)")

    # Compare
    out_bf16 = uint16_to_bf16(final_u16).reshape(height, width, out_channels)
    aie_nchw = out_bf16.float().permute(2, 0, 1).unsqueeze(0)

    print(f"\nAIE output range: [{aie_nchw.min():.4f}, {aie_nchw.max():.4f}]")

    diff = torch.abs(ref_output.float() - aie_nchw)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nComparison (PyTorch vs AIE Composed):")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")

    tolerance = 0.50
    if max_diff < tolerance:
        print(f"  ✓ PASS (max diff < {tolerance})")
        return True
    else:
        print(f"  ✗ FAIL (max diff >= {tolerance})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test RepNCSPELAN (composed)")
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--in-channels", type=int, default=32)
    parser.add_argument("--out-channels", type=int, default=32)
    parser.add_argument("--part-channels", type=int, default=32)
    parser.add_argument("--process-channels", type=int, default=None)
    parser.add_argument("--use-aie", action="store_true")

    args = parser.parse_args()
    success = test_repncsp_elan_composed(
        args.height, args.width, args.in_channels, args.out_channels,
        args.part_channels, args.process_channels, use_aie=args.use_aie,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

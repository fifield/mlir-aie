#!/usr/bin/env python3
"""Test RepNCSPELAN at model dimensions via fully decomposed tiled execution.

RepConv sub-layers run on host CPU (unfused activation order).
Conv+BN+SiLU sub-layers run on NPU via tiled fused kernels.
"""
import sys, os, time, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../python"))
import torch
from mdv6.layers import RepNCSPELAN
import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "elan"))
from test_tiled import run_tiled_fused_conv, fuse_bn, bf16_to_uint16, uint16_to_bf16


def run_repncsp_tiled(repncsp_module, input_hwc, H, W, ic, oc,
                       kh_rn_c1, kh_rn_c3, kh_rn_merge,
                       tile_c1, ocb_c1, tile_c3, ocb_c3, tile_m, ocb_m):
    """Run RepNCSP using tiled fused conv for simple convs, host CPU for RepConv."""
    neck = int(oc * 0.5)

    # Conv1 (1x1, ic→neck) on NPU
    x1 = run_tiled_fused_conv(kh_rn_c1, input_hwc, fuse_bn(repncsp_module.conv1),
                               H, W, neck, tile_c1, tile_c1, ocb_c1,
                               stride=1, kernel_size=1, padding=0)

    # Bottleneck chain — RepConv on host CPU, Conv2 on NPU
    current = x1
    for bn_block in repncsp_module.bottleneck:
        residual = current.clone()
        # RepConv on host (needs unfused add→silu)
        nchw_in = current.float().permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16)
        with torch.no_grad():
            repconv_out_nchw = bn_block.conv1(nchw_in)
        repconv_out = repconv_out_nchw.squeeze(0).permute(1, 2, 0).contiguous()

        # Conv2 (3x3, neck→neck) on NPU
        conv2_out = run_tiled_fused_conv(kh_rn_c3, repconv_out, fuse_bn(bn_block.conv2),
                                          H, W, neck, tile_c3, tile_c3, ocb_c3,
                                          stride=1, kernel_size=3, padding=1)
        current = (residual + conv2_out) if bn_block.residual else conv2_out

    # Conv2 bypass (1x1, ic→neck) on NPU
    x2 = run_tiled_fused_conv(kh_rn_c1, input_hwc, fuse_bn(repncsp_module.conv2),
                               H, W, neck, tile_c1, tile_c1, ocb_c1,
                               stride=1, kernel_size=1, padding=0)

    # Concat + Conv3 merge (1x1, 2*neck→oc) on NPU
    concat = torch.cat([current, x2], dim=2)
    output = run_tiled_fused_conv(kh_rn_merge, concat, fuse_bn(repncsp_module.conv3),
                                   H, W, oc, tile_m, tile_m, ocb_m,
                                   stride=1, kernel_size=1, padding=0)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--in-channels", type=int, default=128)
    parser.add_argument("--out-channels", type=int, default=128)
    parser.add_argument("--part-channels", type=int, default=128)
    parser.add_argument("--process-channels", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    H, W = args.height, args.width
    ic, oc = args.in_channels, args.out_channels
    part_ch, proc_ch = args.part_channels, args.process_channels
    half_part = part_ch // 2

    print(f"\nRepNCSPELAN({ic}→{oc}, part={part_ch}, proc={proc_ch}, r={args.repeat}) at {H}×{W}")

    layer = RepNCSPELAN(ic, oc, part_ch, proc_ch,
                         csp_args={"repeat_num": args.repeat}).eval().to(torch.bfloat16)
    torch.manual_seed(42)
    x = torch.randn(1, ic, H, W, dtype=torch.bfloat16)
    with torch.no_grad():
        ref = layer(x)
    print(f"PyTorch ref: {ref.shape}, [{ref.min():.4f}, {ref.max():.4f}]")

    bd = os.path.join(os.path.dirname(__file__), "..", "conv", "build")
    kh_conv1 = DefaultNPURuntime.load(NPUKernel(f"{bd}/re4_conv1.xclbin", f"{bd}/re4_conv1.bin"))
    kh_c3x3 = DefaultNPURuntime.load(NPUKernel(f"{bd}/re4_conv3x3.xclbin", f"{bd}/re4_conv3x3.bin"))
    kh_conv4 = DefaultNPURuntime.load(NPUKernel(f"{bd}/re4_conv4.xclbin", f"{bd}/re4_conv4.bin"))
    kh_rn_c1 = DefaultNPURuntime.load(NPUKernel(f"{bd}/re4_rn_conv1x1_64_32.xclbin", f"{bd}/re4_rn_conv1x1_64_32.bin"))
    kh_rn_c3 = DefaultNPURuntime.load(NPUKernel(f"{bd}/re4_rn_conv3x3_32_32.xclbin", f"{bd}/re4_rn_conv3x3_32_32.bin"))
    # RepNCSP merge is 64→64 (2*neck=64 → proc=64), reuse elan conv1 xclbin (1x1 64→64)
    kh_rn_m = DefaultNPURuntime.load(NPUKernel(f"{bd}/tf_elan_conv1.xclbin", f"{bd}/tf_elan_conv1.bin"))

    input_hwc = x.squeeze(0).permute(1, 2, 0).contiguous()
    t0 = time.time()

    # Stage 1: Conv1 (1x1, 128→128)
    print("  Conv1...", end=" ", flush=True)
    conv1_out = run_tiled_fused_conv(kh_conv1, input_hwc, fuse_bn(layer.conv1),
                                      H, W, part_ch, 10, 10, 64, stride=1, kernel_size=1, padding=0)
    print("done")

    x1 = conv1_out[:, :, :half_part]
    x2 = conv1_out[:, :, half_part:]

    # Stage 2: RepNCSP#1 (64→64, r=3)
    print("  RepNCSP#1...", end=" ", flush=True)
    x3_rn = run_repncsp_tiled(layer.conv2[0], x2, H, W, half_part, proc_ch,
                               kh_rn_c1, kh_rn_c3, kh_rn_m, 16, 32, 16, 32, 8, 64)
    print("done")

    # Stage 3: Conv3x3#1 (64→64)
    print("  Conv3x3#1...", end=" ", flush=True)
    x3 = run_tiled_fused_conv(kh_c3x3, x3_rn, fuse_bn(layer.conv2[1]),
                               H, W, proc_ch, 12, 12, 16, stride=1, kernel_size=3, padding=1)
    print("done")

    # Stage 4: RepNCSP#2 (64→64, r=3)
    print("  RepNCSP#2...", end=" ", flush=True)
    x4_rn = run_repncsp_tiled(layer.conv3[0], x3, H, W, proc_ch, proc_ch,
                               kh_rn_c1, kh_rn_c3, kh_rn_m, 16, 32, 16, 32, 8, 64)
    print("done")

    # Stage 5: Conv3x3#2 (64→64)
    print("  Conv3x3#2...", end=" ", flush=True)
    x4 = run_tiled_fused_conv(kh_c3x3, x4_rn, fuse_bn(layer.conv3[1]),
                               H, W, proc_ch, 12, 12, 16, stride=1, kernel_size=3, padding=1)
    print("done")

    # Stage 6: Concat + Conv4 (1x1, 256→128)
    concat = torch.cat([x1, x2, x3, x4], dim=2)
    print("  Conv4...", end=" ", flush=True)
    result = run_tiled_fused_conv(kh_conv4, concat, fuse_bn(layer.conv4),
                                   H, W, oc, 8, 8, 32, stride=1, kernel_size=1, padding=0)
    print("done")

    total = time.time() - t0
    aie = result.float().permute(2, 0, 1).unsqueeze(0)
    diff = torch.abs(ref.float() - aie).max().item()
    mean_diff = torch.abs(ref.float() - aie).mean().item()

    print(f"\n  Total: {total:.1f}s")
    print(f"  AIE range: [{aie.min():.4f}, {aie.max():.4f}]")
    print(f"  Max diff: {diff:.4f}, Mean diff: {mean_diff:.6f}")
    tol = 0.5
    ok = diff < tol
    print(f"  {'PASS' if ok else 'FAIL'} (tolerance {tol})")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

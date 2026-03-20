#!/usr/bin/env python3
"""Test ELAN(64→64) at 160×160 using host-composed tiled fused conv sub-layers."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../python"))
import torch
from mdv6.layers import ELAN
import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime


def bf16_to_uint16(t):
    return t.view(torch.uint16).cpu().numpy()

def uint16_to_bf16(a):
    return torch.from_numpy(a.copy()).view(torch.bfloat16)

def fuse_bn(conv_module):
    """Return [conv_weights, fused_bn_w, fused_bn_b] as packed uint16."""
    eps = conv_module.bn.eps
    gamma = conv_module.bn.weight.data
    beta = conv_module.bn.bias.data
    mean = conv_module.bn.running_mean.data
    var = conv_module.bn.running_var.data
    inv_std = 1.0 / torch.sqrt(var + eps)
    return bf16_to_uint16(torch.cat([
        conv_module.conv.weight.data.flatten(),
        gamma * inv_std,
        beta - gamma * mean * inv_std,
    ]))


def fuse_bn_transposed(conv_module):
    """Return [transposed_conv_weights, fused_bn_w, fused_bn_b] as packed uint16.

    Conv weights are transposed from OC-major [oc][ic] to block layout
    [ic/8][oc/8][8ic][8oc] for contiguous vector loads in the AIE kernel.
    Only supports 1x1 convolutions.
    """
    eps = conv_module.bn.eps
    gamma = conv_module.bn.weight.data
    beta = conv_module.bn.bias.data
    mean = conv_module.bn.running_mean.data
    var = conv_module.bn.running_var.data
    inv_std = 1.0 / torch.sqrt(var + eps)

    # Original weight: (oc, ic, 1, 1) for 1x1 conv
    w = conv_module.conv.weight.data.squeeze(-1).squeeze(-1)  # (oc, ic)
    oc, ic = w.shape

    # Transpose to block layout: [ic/8][oc/8][8ic][8oc]
    w_blocks = w.reshape(oc // 8, 8, ic // 8, 8)  # [oc_blk, 8oc, ic_blk, 8ic]
    w_blocks = w_blocks.permute(2, 0, 3, 1)        # [ic_blk, oc_blk, 8ic, 8oc]
    w_transposed = w_blocks.contiguous().flatten()

    return bf16_to_uint16(torch.cat([
        w_transposed,
        gamma * inv_std,
        beta - gamma * mean * inv_std,
    ]))

def extract_patch(image_hwc, tile_row, tile_col, tile_h, tile_w, stride=1, ks=3, pad=1):
    """Extract input patch for tiled conv."""
    H, W, C = image_hwc.shape
    patch_h = (tile_h - 1) * stride + ks
    patch_w = (tile_w - 1) * stride + ks
    in_start_h = tile_row * tile_h * stride - pad
    in_start_w = tile_col * tile_w * stride - pad
    patch = torch.zeros(patch_h, patch_w, C, dtype=image_hwc.dtype)
    vs_h = max(0, in_start_h); vs_w = max(0, in_start_w)
    ve_h = min(H, in_start_h + patch_h); ve_w = min(W, in_start_w + patch_w)
    po_h = vs_h - in_start_h; po_w = vs_w - in_start_w
    patch[po_h:po_h+(ve_h-vs_h), po_w:po_w+(ve_w-vs_w), :] = image_hwc[vs_h:ve_h, vs_w:ve_w, :]
    return patch

def run_tiled_fused_conv(kernel_handle, input_hwc, weights_uint16,
                          out_h, out_w, out_ch, tile_h, tile_w, oc_block,
                          stride=1, kernel_size=3, padding=1):
    """Run a full tiled fused conv layer, returning output HWC tensor."""
    H, W, C = input_hwc.shape
    tiles_h = (out_h + tile_h - 1) // tile_h
    tiles_w = (out_w + tile_w - 1) // tile_w
    n_oc_blocks = (out_ch + oc_block - 1) // oc_block
    output = torch.zeros(out_h, out_w, out_ch, dtype=torch.bfloat16)
    patch_h = (tile_h - 1) * stride + kernel_size
    patch_w = (tile_w - 1) * stride + kernel_size
    patch_size_raw = patch_h * patch_w * C
    patch_size = patch_size_raw + (patch_size_raw % 2)
    output_tile_size = tile_h * tile_w * oc_block
    conv_wt_size = oc_block * C * kernel_size * kernel_size

    # Unpack full weight array: [all_conv_wts (oc*ic*ks*ks), all_bn_w (oc), all_bn_b (oc)]
    total_conv_wts = out_ch * input_hwc.shape[2] * kernel_size * kernel_size
    all_conv_wts = weights_uint16[:total_conv_wts]
    all_bn_w = weights_uint16[total_conv_wts:total_conv_wts + out_ch]
    all_bn_b = weights_uint16[total_conv_wts + out_ch:total_conv_wts + 2 * out_ch]

    for ocb in range(n_oc_blocks):
        oc_start = ocb * oc_block
        oc_end = min(oc_start + oc_block, out_ch)
        actual_oc = oc_end - oc_start
        # Extract per-block weights: [conv_wts_block, bn_w_block, bn_b_block]
        cw_per_oc = input_hwc.shape[2] * kernel_size * kernel_size
        conv_block = all_conv_wts[oc_start * cw_per_oc:oc_end * cw_per_oc]
        bn_w_block = all_bn_w[oc_start:oc_end]
        bn_b_block = all_bn_b[oc_start:oc_end]
        wt_block = np.concatenate([conv_block, bn_w_block, bn_b_block])
        # Pad to expected size if needed
        expected = conv_wt_size + 2 * oc_block
        if len(wt_block) < expected:
            wt_block = np.pad(wt_block, (0, expected - len(wt_block)))

        for tr in range(tiles_h):
            for tc in range(tiles_w):
                patch = extract_patch(input_hwc, tr, tc, tile_h, tile_w,
                                       stride, kernel_size, padding)
                patch_u16 = bf16_to_uint16(patch.flatten())
                if len(patch_u16) < patch_size:
                    patch_u16 = np.pad(patch_u16, (0, patch_size - len(patch_u16)))

                in1 = iron.tensor(patch_u16, dtype=np.uint16)
                in2 = iron.tensor(wt_block, dtype=np.uint16)
                out = iron.zeros(output_tile_size, dtype=np.uint16)
                DefaultNPURuntime.run(kernel_handle, [in1, in2, out])
                tile_out = uint16_to_bf16(out.numpy()[:output_tile_size].copy())
                tile_out = tile_out.reshape(tile_h, tile_w, oc_block)

                oh_s = tr * tile_h; ow_s = tc * tile_w
                oh_e = min(oh_s + tile_h, out_h); ow_e = min(ow_s + tile_w, out_w)
                output[oh_s:oh_e, ow_s:ow_e, oc_start:oc_end] = \
                    tile_out[:oh_e-oh_s, :ow_e-ow_s, :actual_oc]
    return output


def main():
    H, W = 160, 160
    ic, oc = 64, 64
    part_ch = 64
    proc_ch = 32

    print(f"\nTesting ELAN({ic}→{oc}) at {H}×{W} on NPU (host-composed tiled fused)")

    layer = ELAN(ic, oc, part_ch, proc_ch).eval().to(torch.bfloat16)
    torch.manual_seed(42)
    x = torch.randn(1, ic, H, W, dtype=torch.bfloat16)
    with torch.no_grad():
        ref = layer(x)
    print(f"PyTorch ref: {ref.shape}, range [{ref.min():.4f}, {ref.max():.4f}]")

    bd = os.path.join(os.path.dirname(__file__), "..", "conv", "build")

    # Load kernel handles
    kh_conv1 = DefaultNPURuntime.load(NPUKernel(f"{bd}/tf_elan_conv1.xclbin", f"{bd}/tf_elan_conv1.bin"))
    kh_conv3 = DefaultNPURuntime.load(NPUKernel(f"{bd}/tf_elan_conv3x3.xclbin", f"{bd}/tf_elan_conv3x3.bin"))
    kh_conv4 = DefaultNPURuntime.load(NPUKernel(f"{bd}/tf_elan_conv4.xclbin", f"{bd}/tf_elan_conv4.bin"))

    input_hwc = x.squeeze(0).permute(1, 2, 0).contiguous()
    t0 = time.time()

    # Stage 1: Conv1 (1x1, 64→64)
    print("  Conv1 (1x1 64→64)...", end=" ", flush=True)
    wts1 = fuse_bn(layer.conv1)
    conv1_out = run_tiled_fused_conv(kh_conv1, input_hwc, wts1,
                                      H, W, part_ch, 8, 8, 64, stride=1, kernel_size=1, padding=0)
    print("done")

    # Split: x1=first 32ch, x2=last 32ch
    x1 = conv1_out[:, :, :proc_ch]
    x2 = conv1_out[:, :, proc_ch:]

    # Stage 2: Conv2 (3x3, 32→32) on x2
    print("  Conv2 (3x3 32→32)...", end=" ", flush=True)
    wts2 = fuse_bn(layer.conv2)
    x3 = run_tiled_fused_conv(kh_conv3, x2, wts2,
                               H, W, proc_ch, 16, 16, 32, stride=1, kernel_size=3, padding=1)
    print("done")

    # Stage 3: Conv3 (3x3, 32→32) on x3
    print("  Conv3 (3x3 32→32)...", end=" ", flush=True)
    wts3 = fuse_bn(layer.conv3)
    x4 = run_tiled_fused_conv(kh_conv3, x3, wts3,
                               H, W, proc_ch, 16, 16, 32, stride=1, kernel_size=3, padding=1)
    print("done")

    # Stage 4: Concat [x1, x2, x3, x4] → Conv4 (1x1, 128→64)
    concat = torch.cat([x1, x2, x3, x4], dim=2)  # 128ch
    print("  Conv4 (1x1 128→64)...", end=" ", flush=True)
    wts4 = fuse_bn(layer.conv4)
    result = run_tiled_fused_conv(kh_conv4, concat, wts4,
                                   H, W, oc, 8, 8, 64, stride=1, kernel_size=1, padding=0)
    print("done")

    total = time.time() - t0
    print(f"\n  Total time: {total:.1f}s")

    aie_nchw = result.float().permute(2, 0, 1).unsqueeze(0)
    print(f"AIE output range: [{aie_nchw.min():.4f}, {aie_nchw.max():.4f}]")

    diff = torch.abs(ref.float() - aie_nchw)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    tol = 0.5
    if max_diff < tol:
        print(f"✓ PASS (max diff < {tol})")
    else:
        print(f"✗ FAIL (max diff >= {tol})")
    return max_diff < tol


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

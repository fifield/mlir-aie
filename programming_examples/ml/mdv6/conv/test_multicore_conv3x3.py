#!/usr/bin/env python3
"""Test 32-core conv3x3+BN+SiLU with weight broadcast."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../python"))
import torch
import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime


def bf16_to_uint16(t):
    return t.view(torch.uint16).cpu().numpy()

def uint16_to_bf16(a):
    return torch.from_numpy(a.copy()).view(torch.bfloat16)

def fast_sigmoid(x):
    return 0.5 + 0.5 * x / (1.0 + torch.abs(x))

def conv3x3_bn_silu_ref(patch_hwc, conv_w, bn_w, bn_b, tile_h, tile_w, ic, oc, stride, pad):
    """Reference Conv3x3+BN+SiLU in float32."""
    patch_h = (tile_h - 1) * stride + 3
    patch_w = (tile_w - 1) * stride + 3
    out = torch.zeros(tile_h, tile_w, oc)
    conv_w_f = conv_w.float().reshape(oc, ic, 3, 3)
    for oh in range(tile_h):
        for ow in range(tile_w):
            for j in range(oc):
                val = 0.0
                for i in range(ic):
                    for kh in range(3):
                        for kw in range(3):
                            ih = oh * stride + kh
                            iw = ow * stride + kw
                            val += float(patch_hwc[ih, iw, i]) * float(conv_w_f[j, i, kh, kw])
                # BN + SiLU
                val = float(bn_w[j]) * val + float(bn_b[j])
                sig = 0.5 + 0.5 * val / (1.0 + abs(val))
                out[oh, ow, j] = val * sig
    return out.to(torch.bfloat16)


def main():
    tile_h, tile_w = 8, 8
    ic, oc = 16, 16
    n_cores = 32
    stride, padding = 1, 1
    ks = 3

    print(f"\nTest: {n_cores}-core conv3x3 ({ic}->{oc}), tile {tile_h}x{tile_w}")

    bd = os.path.join(os.path.dirname(__file__), "build")
    xclbin = os.path.join(bd, "mc32_3x3.xclbin")
    insts = os.path.join(bd, "mc32_3x3.bin")

    if not os.path.exists(xclbin):
        print(f"ERROR: {xclbin} not found")
        return False

    patch_h = (tile_h - 1) * stride + ks
    patch_w = (tile_w - 1) * stride + ks
    patch_size_raw = patch_h * patch_w * ic
    patch_size = patch_size_raw + (patch_size_raw % 2)
    output_tile_size = tile_h * tile_w * oc
    conv_weight_size = oc * ic * ks * ks
    weight_block_size = conv_weight_size + 2 * oc

    torch.manual_seed(42)
    n_test = min(4, n_cores)  # Only validate a few cores
    patches = []
    for _ in range(n_cores):
        patches.append(torch.randn(patch_h, patch_w, ic, dtype=torch.bfloat16))

    conv_w = torch.randn(oc, ic, ks, ks, dtype=torch.bfloat16) * 0.1
    bn_w = torch.ones(oc, dtype=torch.bfloat16)
    bn_b = torch.zeros(oc, dtype=torch.bfloat16)

    # Pack weights
    weights_u16 = bf16_to_uint16(torch.cat([conv_w.flatten(), bn_w, bn_b]))

    # Reference for first few cores
    refs = []
    for i in range(n_test):
        ref = conv3x3_bn_silu_ref(patches[i], conv_w, bn_w, bn_b,
                                   tile_h, tile_w, ic, oc, stride, padding)
        refs.append(ref)
    print(f"  Reference computed for {n_test} patches")

    # Pack input
    input_parts = []
    for p in patches:
        p_u16 = bf16_to_uint16(p.flatten())
        if len(p_u16) < patch_size:
            p_u16 = np.pad(p_u16, (0, patch_size - len(p_u16)))
        input_parts.append(p_u16)
    input_concat = np.concatenate(input_parts)

    kh = DefaultNPURuntime.load(NPUKernel(xclbin, insts))
    in_buf = iron.tensor(input_concat, dtype=np.uint16)
    wt_buf = iron.tensor(weights_u16, dtype=np.uint16)
    out_buf = iron.zeros(n_cores * output_tile_size, dtype=np.uint16)

    print("  Running on NPU...", end=" ", flush=True)
    t0 = time.time()
    DefaultNPURuntime.run(kh, [in_buf, wt_buf, out_buf])
    elapsed = time.time() - t0
    print(f"done ({elapsed*1000:.1f}ms)")

    out_data = out_buf.numpy().copy()
    max_diff = 0.0
    for c in range(n_test):
        start = c * output_tile_size
        tile_out = uint16_to_bf16(out_data[start:start+output_tile_size])
        tile_out = tile_out.reshape(tile_h, tile_w, oc)
        diff = torch.abs(refs[c].float() - tile_out.float())
        md = diff.max().item()
        print(f"  Core {c}: max_diff={md:.6f}")
        max_diff = max(max_diff, md)

    tol = 0.5
    if max_diff < tol:
        print(f"PASS (max_diff={max_diff:.6f} < {tol})")
        return True
    else:
        print(f"FAIL (max_diff={max_diff:.6f} >= {tol})")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

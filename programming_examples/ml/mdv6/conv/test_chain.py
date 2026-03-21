#!/usr/bin/env python3
"""Level 4 test: 2-operator chain through memtile.
Verifies Conv1x1→Conv1x1 pipeline produces same result as sequential execution.
Both convs use the same weights (same ch→ch dims), data stays on-chip between them."""
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
    """Fast sigmoid matching AIE kernel: 0.5 + 0.5*x/(1+|x|)"""
    return 0.5 + 0.5 * x / (1.0 + torch.abs(x))

def conv1x1_bn_silu_ref(x_hwc, conv_w, bn_w, bn_b):
    """Reference Conv1x1+BN+SiLU using fast sigmoid (matches AIE kernel)."""
    H, W, ic = x_hwc.shape
    oc = conv_w.shape[0]
    x_flat = x_hwc.reshape(-1, ic).float()
    out = x_flat @ conv_w.float().t()
    out = out * bn_w.float().unsqueeze(0) + bn_b.float().unsqueeze(0)
    out = out * fast_sigmoid(out)
    return out.to(torch.bfloat16).reshape(H, W, oc)


def main():
    tile_h, tile_w = 8, 8
    ch = 16

    print(f"\nLevel 4: chain Conv1x1({ch}→{ch}) x2, tile {tile_h}x{tile_w}")

    bd = os.path.join(os.path.dirname(__file__), "build")
    xclbin = os.path.join(bd, "chain.xclbin")
    insts = os.path.join(bd, "chain.bin")

    if not os.path.exists(xclbin):
        print(f"ERROR: {xclbin} not found. Build first.")
        return False

    # Create random test data
    torch.manual_seed(42)
    x = torch.randn(tile_h, tile_w, ch, dtype=torch.bfloat16)
    conv_w = torch.randn(ch, ch, dtype=torch.bfloat16)
    bn_w = torch.randn(ch, dtype=torch.bfloat16) * 0.5 + 1.0
    bn_b = torch.randn(ch, dtype=torch.bfloat16) * 0.1

    # Pack weights: [conv_weights(ch*ch), bn_w(ch), bn_b(ch)]
    weights_packed = torch.cat([conv_w.flatten(), bn_w, bn_b])
    weights_u16 = bf16_to_uint16(weights_packed)

    # Reference: Conv1 → Conv2 (same weights for both)
    inter = conv1x1_bn_silu_ref(x, conv_w, bn_w, bn_b)
    ref = conv1x1_bn_silu_ref(inter, conv_w, bn_w, bn_b)
    print(f"  Reference: input [{x.min():.3f}, {x.max():.3f}] → "
          f"inter [{inter.min():.3f}, {inter.max():.3f}] → "
          f"output [{ref.min():.3f}, {ref.max():.3f}]")

    data_size = tile_h * tile_w * ch
    input_u16 = bf16_to_uint16(x.flatten())

    # Load and run on NPU
    kh = DefaultNPURuntime.load(NPUKernel(xclbin, insts))

    in_buf = iron.tensor(input_u16, dtype=np.uint16)
    wt_buf = iron.tensor(weights_u16, dtype=np.uint16)
    out_buf = iron.zeros(data_size, dtype=np.uint16)

    print("  Running on NPU...", end=" ", flush=True)
    t0 = time.time()
    DefaultNPURuntime.run(kh, [in_buf, wt_buf, out_buf])
    elapsed = time.time() - t0
    print(f"done ({elapsed*1000:.1f}ms)")

    # Compare
    output_bf16 = uint16_to_bf16(out_buf.numpy()[:data_size].copy())
    output = output_bf16.reshape(tile_h, tile_w, ch)

    diff = torch.abs(ref.float() - output.float())
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    # With 2 chained conv+BN+SiLU, error accumulates
    tol = 1.0
    if max_diff < tol:
        print(f"PASS (max_diff={max_diff:.6f} < {tol})")
        return True
    else:
        print(f"FAIL (max_diff={max_diff:.6f} >= {tol})")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

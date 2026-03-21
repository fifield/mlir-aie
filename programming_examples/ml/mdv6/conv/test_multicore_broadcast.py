#!/usr/bin/env python3
"""Level 2 test: 2-tile weight broadcast.
Verifies 2 cores processing independent spatial patches with shared weights
produce the same result as running each patch through the single-core reference."""
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
    # SiLU with fast sigmoid (matches AIE: x * (0.5 + x/(2+2*|x|)))
    out = out * fast_sigmoid(out)
    return out.to(torch.bfloat16).reshape(H, W, oc)


def main():
    tile_h, tile_w = 8, 8
    ic, oc = 16, 16
    n_cores = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    print(f"\nLevel {2 if n_cores <= 2 else 3}: {n_cores}-core broadcast conv1x1 ({ic}→{oc}), tile {tile_h}x{tile_w}")

    bd = os.path.join(os.path.dirname(__file__), "build")
    xclbin = os.path.join(bd, f"mc_{n_cores}core.xclbin")
    insts = os.path.join(bd, f"mc_{n_cores}core.bin")
    # Fallback for 2-core
    if not os.path.exists(xclbin) and n_cores == 2:
        xclbin = os.path.join(bd, "mc_broadcast.xclbin")
        insts = os.path.join(bd, "mc_broadcast.bin")

    if not os.path.exists(xclbin):
        print(f"ERROR: {xclbin} not found. Build first.")
        return False

    # Create random test data
    torch.manual_seed(42)
    patches = [torch.randn(tile_h, tile_w, ic, dtype=torch.bfloat16) for _ in range(n_cores)]
    conv_w = torch.randn(oc, ic, dtype=torch.bfloat16)
    bn_w = torch.randn(oc, dtype=torch.bfloat16) * 0.5 + 1.0
    bn_b = torch.randn(oc, dtype=torch.bfloat16) * 0.1

    # Fuse BN params
    eps = 1e-5
    # For reference, simulate: fused_bn_w = gamma/sqrt(var+eps), fused_bn_b = beta - gamma*mean/sqrt(var+eps)
    # Since we're testing without actual BN running stats, use bn_w directly as fused_w, bn_b as fused_b
    fused_w = bn_w
    fused_b = bn_b

    # Pack weights: [conv_weights(oc*ic), bn_w(oc), bn_b(oc)]
    weights_packed = torch.cat([conv_w.flatten(), fused_w, fused_b])
    weights_u16 = bf16_to_uint16(weights_packed)

    # Reference: run each patch independently
    refs = []
    for p in patches:
        ref = conv1x1_bn_silu_ref(p, conv_w, fused_w, fused_b)
        refs.append(ref)
    print(f"  Reference computed for {n_cores} patches")

    # Pack input: concatenate all patches
    input_concat = torch.cat([p.flatten() for p in patches])
    input_u16 = bf16_to_uint16(input_concat)

    patch_size = tile_h * tile_w * ic
    output_tile_size = tile_h * tile_w * oc

    # Load and run on NPU
    kh = DefaultNPURuntime.load(NPUKernel(xclbin, insts))

    in_buf = iron.tensor(input_u16, dtype=np.uint16)
    wt_buf = iron.tensor(weights_u16, dtype=np.uint16)
    out_buf = iron.zeros(n_cores * output_tile_size, dtype=np.uint16)

    print("  Running on NPU...", end=" ", flush=True)
    t0 = time.time()
    DefaultNPURuntime.run(kh, [in_buf, wt_buf, out_buf])
    elapsed = time.time() - t0
    print(f"done ({elapsed*1000:.1f}ms)")

    # Compare results
    output_u16 = out_buf.numpy().copy()
    max_diff = 0.0
    for c in range(n_cores):
        start = c * output_tile_size
        end = start + output_tile_size
        tile_out = uint16_to_bf16(output_u16[start:end]).reshape(tile_h, tile_w, oc)
        diff = torch.abs(refs[c].float() - tile_out.float())
        md = diff.max().item()
        print(f"  Core {c}: max_diff={md:.6f}, mean_diff={diff.mean().item():.6f}")
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

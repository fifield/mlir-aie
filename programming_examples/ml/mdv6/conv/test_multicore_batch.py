#!/usr/bin/env python3
"""Test 32-core conv1x1 with patches_per_core=4 (128 tiles per invocation)."""
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


def main():
    tile_h, tile_w = 8, 8
    ic, oc = 16, 16
    n_cores = 32
    ppc = 4  # patches per core
    total = n_cores * ppc  # 128 tiles per invocation

    print(f"\nTest: {n_cores}-core conv1x1, {ppc} patches/core = {total} tiles/invocation")

    bd = os.path.join(os.path.dirname(__file__), "build")
    xclbin = os.path.join(bd, "mc32_1x1_4ppc.xclbin")
    insts = os.path.join(bd, "mc32_1x1_4ppc.bin")

    if not os.path.exists(xclbin):
        print(f"ERROR: {xclbin} not found")
        return False

    patch_size = tile_h * tile_w * ic
    output_tile_size = tile_h * tile_w * oc

    torch.manual_seed(42)
    # Generate random input for all tiles
    all_input = torch.randn(total * tile_h * tile_w, ic, dtype=torch.bfloat16)
    conv_w = torch.randn(oc, ic, dtype=torch.bfloat16) * 0.1
    bn_w = torch.ones(oc, dtype=torch.bfloat16)
    bn_b = torch.zeros(oc, dtype=torch.bfloat16)

    # Reference: Conv1x1 + BN + SiLU on all tiles
    ref_flat = all_input.float() @ conv_w.float().t()
    ref_flat = ref_flat * bn_w.float().unsqueeze(0) + bn_b.float().unsqueeze(0)
    ref_flat = ref_flat * fast_sigmoid(ref_flat)
    ref_flat = ref_flat.to(torch.bfloat16)

    weights_u16 = bf16_to_uint16(torch.cat([conv_w.flatten(), bn_w, bn_b]))
    input_u16 = bf16_to_uint16(all_input.flatten())

    kh = DefaultNPURuntime.load(NPUKernel(xclbin, insts))
    in_buf = iron.tensor(input_u16, dtype=np.uint16)
    wt_buf = iron.tensor(weights_u16, dtype=np.uint16)
    out_buf = iron.zeros(total * output_tile_size, dtype=np.uint16)

    print("  Running on NPU...", end=" ", flush=True)
    t0 = time.time()
    DefaultNPURuntime.run(kh, [in_buf, wt_buf, out_buf])
    elapsed = time.time() - t0
    print(f"done ({elapsed*1000:.1f}ms)")
    print(f"  Throughput: {total / elapsed:.0f} tiles/s, {total * tile_h * tile_w / elapsed:.0f} pixels/s")

    out_data = uint16_to_bf16(out_buf.numpy()[:total * output_tile_size].copy())
    out_data = out_data.reshape(-1, oc)
    diff = torch.abs(ref_flat.float() - out_data.float())
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    tol = 0.5
    if max_diff < tol:
        print(f"PASS (max_diff={max_diff:.6f} < {tol})")
        return True
    else:
        print(f"FAIL (max_diff={max_diff:.6f} >= {tol})")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

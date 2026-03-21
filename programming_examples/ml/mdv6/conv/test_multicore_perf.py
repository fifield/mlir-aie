#!/usr/bin/env python3
"""Performance test: 32-core conv1x1 at model dimensions.
Measures wall-clock time for processing an 80x80 layer with 16 channels,
comparing single invocation (32 patches) throughput."""
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

def conv1x1_bn_silu_ref(x_hwc, conv_w, bn_w, bn_b):
    H, W, ic = x_hwc.shape
    oc = conv_w.shape[0]
    x_flat = x_hwc.reshape(-1, ic).float()
    out = x_flat @ conv_w.float().t()
    out = out * bn_w.float().unsqueeze(0) + bn_b.float().unsqueeze(0)
    out = out * fast_sigmoid(out)
    return out.to(torch.bfloat16).reshape(H, W, oc)


def main():
    tile_h, tile_w = 10, 10
    ic, oc = 16, 16
    n_cores = 32

    # Target: 80x80 layer with 16 channels
    target_h, target_w = 80, 80
    tiles_h = target_h // tile_h  # 8
    tiles_w = target_w // tile_w  # 8
    total_tiles = tiles_h * tiles_w  # 64
    invocations = (total_tiles + n_cores - 1) // n_cores  # ceil(64/32) = 2

    print(f"\nPerf test: {n_cores}-core conv1x1 ({ic}->{oc}), tile {tile_h}x{tile_w}")
    print(f"  Target layer: {target_h}x{target_w}x{ic} -> {target_h}x{target_w}x{oc}")
    print(f"  Total tiles: {total_tiles}, invocations: {invocations}")

    bd = os.path.join(os.path.dirname(__file__), "build")
    xclbin = os.path.join(bd, f"mc_{n_cores}core_{tile_h}x{tile_w}.xclbin")
    insts = os.path.join(bd, f"mc_{n_cores}core_{tile_h}x{tile_w}.bin")

    if not os.path.exists(xclbin):
        print(f"ERROR: {xclbin} not found. Build first.")
        return False

    # Random test data
    torch.manual_seed(42)
    full_input = torch.randn(target_h, target_w, ic, dtype=torch.bfloat16)
    conv_w = torch.randn(oc, ic, dtype=torch.bfloat16) * 0.1
    bn_w = torch.ones(oc, dtype=torch.bfloat16)
    bn_b = torch.zeros(oc, dtype=torch.bfloat16)

    # Reference
    ref = conv1x1_bn_silu_ref(full_input, conv_w, bn_w, bn_b)

    # Pack weights
    weights_u16 = bf16_to_uint16(torch.cat([conv_w.flatten(), bn_w, bn_b]))

    patch_size = tile_h * tile_w * ic
    output_tile_size = tile_h * tile_w * oc

    kh = DefaultNPURuntime.load(NPUKernel(xclbin, insts))

    # Process all tiles
    output = torch.zeros(target_h, target_w, oc, dtype=torch.bfloat16)
    tile_idx = 0

    t0 = time.time()
    for inv in range(invocations):
        patches_this_inv = min(n_cores, total_tiles - tile_idx)

        # Build input buffer: concatenate patches for this invocation
        patches_u16 = []
        tile_coords = []
        for i in range(patches_this_inv):
            tr = tile_idx // tiles_w
            tc = tile_idx % tiles_w
            tile_coords.append((tr, tc))
            oh_s, ow_s = tr * tile_h, tc * tile_w
            patch = full_input[oh_s:oh_s+tile_h, ow_s:ow_s+tile_w, :]
            patches_u16.append(bf16_to_uint16(patch.flatten()))
            tile_idx += 1

        # Pad to n_cores patches if needed
        while len(patches_u16) < n_cores:
            patches_u16.append(np.zeros(patch_size, dtype=np.uint16))

        input_concat = np.concatenate(patches_u16)
        in_buf = iron.tensor(input_concat, dtype=np.uint16)
        wt_buf = iron.tensor(weights_u16, dtype=np.uint16)
        out_buf = iron.zeros(n_cores * output_tile_size, dtype=np.uint16)

        DefaultNPURuntime.run(kh, [in_buf, wt_buf, out_buf])

        # Unpack results
        out_data = out_buf.numpy().copy()
        for i in range(patches_this_inv):
            tr, tc = tile_coords[i]
            oh_s, ow_s = tr * tile_h, tc * tile_w
            start = i * output_tile_size
            tile_out = uint16_to_bf16(out_data[start:start+output_tile_size])
            output[oh_s:oh_s+tile_h, ow_s:ow_s+tile_w, :] = tile_out.reshape(tile_h, tile_w, oc)

    elapsed = time.time() - t0

    # Compare
    diff = torch.abs(ref.float() - output.float())
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\n  Wall time: {elapsed*1000:.1f}ms ({invocations} invocations)")
    print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
    print(f"  Throughput: {total_tiles / elapsed:.0f} tiles/s")

    tol = 0.5
    if max_diff < tol:
        print(f"PASS (max_diff={max_diff:.6f} < {tol})")
        return True
    else:
        print(f"FAIL (max_diff={max_diff:.6f} >= {tol})")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

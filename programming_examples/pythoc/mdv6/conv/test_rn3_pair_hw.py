#!/usr/bin/env python3
"""Hardware smoke for the tiny scalar rn3-pair fused prototype."""
import os
import sys
from pathlib import Path
import numpy as np
import torch
import pyxrt as xrt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_rn3_pair  # noqa: E402
from rn3_pair_layout import block_major_to_hwc, hwc_to_block_major  # noqa: E402


def f32_to_bf16_u16(a):
    t = torch.from_numpy(np.asarray(a, dtype=np.float32)).to(torch.bfloat16)
    return t.view(torch.uint16).cpu().numpy().copy()


def bf16_u16_to_f32(a):
    return torch.from_numpy(np.asarray(a, dtype=np.uint16).copy()).view(torch.bfloat16).float().numpy()


def silu_approx(x):
    ax = np.abs(x)
    return x * (0.5 + x / (2.0 + 2.0 * ax))


def cpu_kernel_oracle(input_patch, weights, tile_h=8, tile_w=8, ic=4, mid=4, ocb=4):
    w1_size = mid * ic * 9
    bn1_w_off = w1_size
    bn1_b_off = bn1_w_off + mid
    w2_off = bn1_b_off + mid
    w2_size = ocb * mid * 9
    bn2_w_off = w2_off + w2_size
    bn2_b_off = bn2_w_off + ocb
    out = np.zeros((tile_h, tile_w, ocb), dtype=np.float32)
    patch_w = tile_w + 4
    flat = input_patch.reshape(-1)
    for r in range(tile_h):
        for c in range(tile_w):
            for o2 in range(ocb):
                acc2 = np.float32(0.0)
                for kh2 in range(3):
                    for kw2 in range(3):
                        mr = r + kh2
                        mc = c + kw2
                        for o1 in range(mid):
                            acc1 = np.float32(0.0)
                            for kh1 in range(3):
                                for kw1 in range(3):
                                    for i in range(ic):
                                        in_idx = ((mr + kh1) * patch_w + (mc + kw1)) * ic + i
                                        w1_idx = ((o1 * ic + i) * 3 + kh1) * 3 + kw1
                                        acc1 += flat[in_idx] * weights[w1_idx]
                            x1 = acc1 * weights[bn1_w_off + o1] + weights[bn1_b_off + o1]
                            y1 = silu_approx(x1)
                            w2_idx = w2_off + ((o2 * mid + o1) * 3 + kh2) * 3 + kw2
                            acc2 += y1 * weights[w2_idx]
                x2 = acc2 * weights[bn2_w_off + o2] + weights[bn2_b_off + o2]
                out[r, c, o2] = silu_approx(x2)
    return out


def _bo_fill(bo, arr):
    mv = bo.map()
    np.copyto(np.frombuffer(mv, dtype=np.uint8, count=arr.nbytes), np.frombuffer(arr, dtype=np.uint8), casting="no")
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)


def _bo_read(bo, nelem):
    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    return np.frombuffer(bo.map(), dtype=np.uint16, count=nelem).copy()


def test_rn3_pair_npu_matches_cpu_oracle(layer="tiny", check_hwc=False):
    elf = build_rn3_pair.build_one(layer)
    rng = np.random.default_rng(7)
    shape = tuple(x for x in build_rn3_pair._LAYERS[layer] if isinstance(x, int))
    multioc = "multioc" in layer
    if multioc:
        if len(shape) == 7:
            tile_h, tile_w, ic, mid, ocb, n_ocb, n_patches = shape
        else:
            tile_h, tile_w, ic, mid, ocb, n_ocb = shape
            n_patches = 1
        n_cores = 1
        input_patches = rng.normal(0, 0.15, size=(n_patches, tile_h + 4, tile_w + 4, ic)).astype(np.float32)
        weight_blocks = []
        expected_blocks = []
        # Build one independent per-OC-block weight slice. Conv1 is duplicated
        # per block in this bring-up layout; production should share/stream it.
        for _ in range(n_ocb):
            w1 = rng.normal(0, 0.05, size=(mid, ic, 3, 3)).astype(np.float32)
            bn1w = rng.normal(1.0, 0.02, size=(mid,)).astype(np.float32)
            bn1b = rng.normal(0.0, 0.01, size=(mid,)).astype(np.float32)
            w2 = rng.normal(0, 0.05, size=(ocb, mid, 3, 3)).astype(np.float32)
            bn2w = rng.normal(1.0, 0.02, size=(ocb,)).astype(np.float32)
            bn2b = rng.normal(0.0, 0.01, size=(ocb,)).astype(np.float32)
            weight_blocks.append(np.concatenate([w1.reshape(-1), bn1w, bn1b, w2.reshape(-1), bn2w, bn2b]).astype(np.float32))
        weights = np.concatenate(weight_blocks).astype(np.float32)
        in_u16 = f32_to_bf16_u16(input_patches.reshape(-1))
        wt_u16 = f32_to_bf16_u16(weights)
        input_bf = bf16_u16_to_f32(in_u16).reshape(input_patches.shape)
        weights_bf = bf16_u16_to_f32(wt_u16)
        block_len = weight_blocks[0].size
        expected = np.stack([
            np.stack([
                cpu_kernel_oracle(input_bf[p], weights_bf[i * block_len:(i + 1) * block_len], tile_h, tile_w, ic, mid, ocb)
                for i in range(n_ocb)
            ], axis=0)
            for p in range(n_patches)
        ], axis=0)
        out_nelem = n_patches * n_ocb * tile_h * tile_w * ocb
        out_shape = (n_patches, n_ocb, tile_h, tile_w, ocb)
    else:
        if len(shape) == 6:
            n_cores, tile_h, tile_w, ic, mid, ocb = shape
        else:
            n_cores = 1
            tile_h, tile_w, ic, mid, ocb = shape
        input_patches = rng.normal(0, 0.15, size=(n_cores, tile_h + 4, tile_w + 4, ic)).astype(np.float32)
        w1 = rng.normal(0, 0.05, size=(mid, ic, 3, 3)).astype(np.float32)
        bn1w = rng.normal(1.0, 0.02, size=(mid,)).astype(np.float32)
        bn1b = rng.normal(0.0, 0.01, size=(mid,)).astype(np.float32)
        w2 = rng.normal(0, 0.05, size=(ocb, mid, 3, 3)).astype(np.float32)
        bn2w = rng.normal(1.0, 0.02, size=(ocb,)).astype(np.float32)
        bn2b = rng.normal(0.0, 0.01, size=(ocb,)).astype(np.float32)
        weights = np.concatenate([w1.reshape(-1), bn1w, bn1b, w2.reshape(-1), bn2w, bn2b]).astype(np.float32)
        in_u16 = f32_to_bf16_u16(input_patches.reshape(-1))
        wt_u16 = f32_to_bf16_u16(weights)
        # Compare against the same bf16-rounded inputs/weights the NPU sees.
        input_bf = bf16_u16_to_f32(in_u16).reshape(input_patches.shape)
        weights_bf = bf16_u16_to_f32(wt_u16)
        expected = np.stack([
            cpu_kernel_oracle(input_bf[i], weights_bf, tile_h, tile_w, ic, mid, ocb)
            for i in range(n_cores)
        ], axis=0)
        out_nelem = n_cores * tile_h * tile_w * ocb
        out_shape = (n_cores, tile_h, tile_w, ocb)

    dev = xrt.device(0)
    kernel = xrt.ext.kernel(xrt.hw_context(dev, xrt.elf(elf)), "main")
    in_bo = xrt.ext.bo(dev, in_u16.nbytes)
    wt_bo = xrt.ext.bo(dev, wt_u16.nbytes)
    out_bo = xrt.ext.bo(dev, out_nelem * 2)
    _bo_fill(in_bo, in_u16)
    _bo_fill(wt_bo, wt_u16)
    r = xrt.run(kernel)
    r.set_arg(0, in_bo)
    r.set_arg(1, wt_bo)
    r.set_arg(2, out_bo)
    r.start()
    r.wait2()
    got = bf16_u16_to_f32(_bo_read(out_bo, out_nelem)).reshape(out_shape)

    max_abs = float(np.max(np.abs(got - expected)))
    print(f"max_abs={max_abs:.6f}")
    np.testing.assert_allclose(got, expected, rtol=2e-2, atol=2e-2)
    if check_hwc:
        if not multioc:
            raise ValueError("--check-hwc is only meaningful for multioc block-major output")
        got_hwc = block_major_to_hwc(got)
        exp_hwc = block_major_to_hwc(expected)
        roundtrip = hwc_to_block_major(got_hwc, ocb)
        np.testing.assert_array_equal(roundtrip, got)
        np.testing.assert_allclose(got_hwc, exp_hwc, rtol=2e-2, atol=2e-2)
        print(f"hwc_shape={got_hwc.shape}")


# Shapes whose (input + weight + output + stack) footprint fits in the
# 64 KB L1 budget for the scalar prototype kernel. Larger shapes
# (re6_tile, re6_oc8) need vectorized weights or K-block-style weight
# streaming — out of scope for this correctness anchor.
_L1_FITTING = ["tiny", "re6_oc4", "re4_tile"]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--layer",
                   choices=list(build_rn3_pair._LAYERS) + ["all"],
                   default="tiny",
                   help="`all` runs every L1-fitting shape")
    p.add_argument("--check-hwc", action="store_true",
                   help="for multioc layers, verify host block-major -> HWC/full-OC conversion")
    args = p.parse_args()
    labels = _L1_FITTING if args.layer == "all" else [args.layer]
    for label in labels:
        test_rn3_pair_npu_matches_cpu_oracle(label, check_hwc=args.check_hwc)
        print(f"PASS: {label} rn3-pair NPU smoke matches CPU oracle")

#!/usr/bin/env python3
"""Full-layer-style rn3pair p4 chunk wrapper smoke test.

This does not wire the fused kernel into the MDV6 runtime yet. It proves the
host wrapper shape needed for that integration:

1. pack a full HWC input image into row-major `(tile+4)` patches,
2. run the selected `re6_oc4_multioc12_p4` ELF on p4 chunks,
3. discard padded final-chunk outputs,
4. convert block-major output to HWC/full-OC,
5. scatter tiles back to a full HWC image,
6. compare against the CPU fused-tile oracle using the same bf16-rounded inputs
   and weights the NPU sees.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pyxrt as xrt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_rn3_pair  # noqa: E402
from rn3_pair_layout import (  # noqa: E402
    block_major_to_hwc,
    iter_patch_chunks,
    pack_rn3_pair_input_patches,
    pad_patch_chunk,
    scatter_tile_hwc_to_image,
)
from test_rn3_pair_hw import (  # noqa: E402
    bf16_u16_to_f32,
    cpu_kernel_oracle,
    f32_to_bf16_u16,
    _bo_fill,
    _bo_read,
)


def make_weight_blocks(rng, ic=48, mid=48, ocb=4, n_ocb=12):
    blocks = []
    for _ in range(n_ocb):
        w1 = rng.normal(0, 0.05, size=(mid, ic, 3, 3)).astype(np.float32)
        bn1w = rng.normal(1.0, 0.02, size=(mid,)).astype(np.float32)
        bn1b = rng.normal(0.0, 0.01, size=(mid,)).astype(np.float32)
        w2 = rng.normal(0, 0.05, size=(ocb, mid, 3, 3)).astype(np.float32)
        bn2w = rng.normal(1.0, 0.02, size=(ocb,)).astype(np.float32)
        bn2b = rng.normal(0.0, 0.01, size=(ocb,)).astype(np.float32)
        blocks.append(np.concatenate([w1.reshape(-1), bn1w, bn1b, w2.reshape(-1), bn2w, bn2b]).astype(np.float32))
    return blocks


def cpu_expected_tiles(patches_bf, weights_bf, block_len, tile_h, tile_w, ic, mid, ocb, n_ocb):
    block_major = np.stack([
        np.stack([
            cpu_kernel_oracle(patches_bf[p], weights_bf[i * block_len:(i + 1) * block_len], tile_h, tile_w, ic, mid, ocb)
            for i in range(n_ocb)
        ], axis=0)
        for p in range(patches_bf.shape[0])
    ], axis=0)
    return block_major_to_hwc(block_major)


def run_p4_chunks_npu(image_hwc, weight_blocks, *, tile_h=8, tile_w=8, ic=48, mid=48, ocb=4, n_ocb=12):
    layer = "re6_oc4_multioc12_p4"
    elf = str(build_rn3_pair.build_one(layer))
    n_patches_per_dispatch = 4
    patches = pack_rn3_pair_input_patches(image_hwc, tile_h, tile_w, halo=2)
    weights = np.concatenate(weight_blocks).astype(np.float32)

    # Match the precision seen by the hardware.
    patches_u16 = f32_to_bf16_u16(patches.reshape(-1))
    weights_u16 = f32_to_bf16_u16(weights)
    patches_bf = bf16_u16_to_f32(patches_u16).reshape(patches.shape)
    weights_bf = bf16_u16_to_f32(weights_u16)

    block_len = weight_blocks[0].size
    expected_tiles_hwc = cpu_expected_tiles(patches_bf, weights_bf, block_len, tile_h, tile_w, ic, mid, ocb, n_ocb)

    dev = xrt.device(0)
    kernel = xrt.ext.kernel(xrt.hw_context(dev, xrt.elf(elf)), "main")
    wt_bo = xrt.ext.bo(dev, weights_u16.nbytes)
    _bo_fill(wt_bo, weights_u16)

    out_nelem = n_patches_per_dispatch * n_ocb * tile_h * tile_w * ocb
    got_tiles = []
    dispatches = 0
    for _start, chunk in iter_patch_chunks(patches_bf, n_patches_per_dispatch):
        padded, valid = pad_patch_chunk(chunk, n_patches_per_dispatch)
        in_u16 = f32_to_bf16_u16(padded.reshape(-1))
        in_bo = xrt.ext.bo(dev, in_u16.nbytes)
        out_bo = xrt.ext.bo(dev, out_nelem * 2)
        _bo_fill(in_bo, in_u16)
        r = xrt.run(kernel)
        r.set_arg(0, in_bo)
        r.set_arg(1, wt_bo)
        r.set_arg(2, out_bo)
        r.start()
        r.wait2()
        got_block = bf16_u16_to_f32(_bo_read(out_bo, out_nelem)).reshape(n_patches_per_dispatch, n_ocb, tile_h, tile_w, ocb)
        got_tiles.append(block_major_to_hwc(got_block)[:valid])
        dispatches += 1

    got_tiles_hwc = np.concatenate(got_tiles, axis=0)
    got_image = scatter_tile_hwc_to_image(got_tiles_hwc, image_hwc.shape[0], image_hwc.shape[1], tile_h, tile_w)
    expected_image = scatter_tile_hwc_to_image(expected_tiles_hwc, image_hwc.shape[0], image_hwc.shape[1], tile_h, tile_w)
    return got_tiles_hwc, expected_tiles_hwc, got_image, expected_image, dispatches


def main():
    rng = np.random.default_rng(11)
    tile_h = tile_w = 8
    ic = mid = 48
    ocb = 4
    n_ocb = 12
    # 24x16 => 6 output tiles, so the p4 wrapper exercises one full chunk and
    # one padded final chunk.
    image = rng.normal(0, 0.15, size=(24, 16, ic)).astype(np.float32)
    weight_blocks = make_weight_blocks(rng, ic, mid, ocb, n_ocb)
    got_tiles, expected_tiles, got_image, expected_image, dispatches = run_p4_chunks_npu(
        image,
        weight_blocks,
        tile_h=tile_h,
        tile_w=tile_w,
        ic=ic,
        mid=mid,
        ocb=ocb,
        n_ocb=n_ocb,
    )
    tile_max = float(np.max(np.abs(got_tiles - expected_tiles)))
    image_max = float(np.max(np.abs(got_image - expected_image)))
    print(f"dispatches={dispatches}")
    print(f"tiles_shape={got_tiles.shape}")
    print(f"image_shape={got_image.shape}")
    print(f"tile_max_abs={tile_max:.6f}")
    print(f"image_max_abs={image_max:.6f}")
    np.testing.assert_allclose(got_tiles, expected_tiles, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(got_image, expected_image, rtol=2e-2, atol=2e-2)
    print("PASS: full-layer p4 rn3pair wrapper matches CPU oracle")


if __name__ == "__main__":
    main()

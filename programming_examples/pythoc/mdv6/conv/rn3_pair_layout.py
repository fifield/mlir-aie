#!/usr/bin/env python3
"""Host-side layout helpers for rn3 pair-fusion bring-up.

The scalar multi-OC NPU prototype currently emits block-major output because
that is the simplest shape for MemTile joins:

    [patch, oc_block, tile_h, tile_w, ocb]

The existing runtime integration path is expected to want per-patch HWC/full-OC
layout:

    [patch, tile_h, tile_w, oc]

These helpers keep that conversion explicit and testable until the device side
writes HWC directly.
"""
from __future__ import annotations

import numpy as np


def block_major_to_hwc(block_major: np.ndarray) -> np.ndarray:
    """Convert [P, OCBLK, H, W, OCB] to [P, H, W, OCBLK*OCB]."""
    a = np.asarray(block_major)
    if a.ndim != 5:
        raise ValueError(f"expected rank-5 block-major array, got shape {a.shape}")
    p, n_ocb, h, w, ocb = a.shape
    return a.transpose(0, 2, 3, 1, 4).reshape(p, h, w, n_ocb * ocb)


def hwc_to_block_major(hwc: np.ndarray, ocb: int) -> np.ndarray:
    """Convert [P, H, W, OC] to [P, OC/OCB, H, W, OCB]."""
    a = np.asarray(hwc)
    if a.ndim != 4:
        raise ValueError(f"expected rank-4 HWC batch, got shape {a.shape}")
    if ocb <= 0 or a.shape[-1] % ocb:
        raise ValueError(f"OC={a.shape[-1]} must be divisible by ocb={ocb}")
    p, h, w, oc = a.shape
    n_ocb = oc // ocb
    return a.reshape(p, h, w, n_ocb, ocb).transpose(0, 3, 1, 2, 4)


def scatter_tile_hwc_to_image(tile_hwc: np.ndarray, image_h: int, image_w: int, tile_h: int, tile_w: int) -> np.ndarray:
    """Scatter [P,H,W,OC] tile outputs row-major into [image_h,image_w,OC].

    Requires that image_h/image_w are exact multiples of tile_h/tile_w. This is
    the simple first integration target; edge tiles can be added after the fused
    path is proven on aligned shapes.
    """
    tiles = np.asarray(tile_hwc)
    if tiles.ndim != 4:
        raise ValueError(f"expected [P,H,W,OC], got {tiles.shape}")
    p, h, w, oc = tiles.shape
    if h != tile_h or w != tile_w:
        raise ValueError(f"tile tensor has tile {(h, w)}, expected {(tile_h, tile_w)}")
    if image_h % tile_h or image_w % tile_w:
        raise ValueError("image dimensions must be exact multiples of tile size")
    grid_h = image_h // tile_h
    grid_w = image_w // tile_w
    if p != grid_h * grid_w:
        raise ValueError(f"got {p} patches, expected {grid_h * grid_w} for image/grid")
    out = np.empty((image_h, image_w, oc), dtype=tiles.dtype)
    idx = 0
    for tr in range(grid_h):
        for tc in range(grid_w):
            out[tr * tile_h:(tr + 1) * tile_h, tc * tile_w:(tc + 1) * tile_w, :] = tiles[idx]
            idx += 1
    return out


def pack_rn3_pair_input_patches(image_hwc: np.ndarray, tile_h: int, tile_w: int, halo: int = 2) -> np.ndarray:
    """Pack row-major `(tile+2*halo)` patches from HWC image with zero padding.

    For 3x3+3x3 pair fusion, halo=2. Each output tile consumes a
    `(tile_h+4) x (tile_w+4)` original-input patch.
    """
    img = np.asarray(image_hwc)
    if img.ndim != 3:
        raise ValueError(f"expected HWC image, got {img.shape}")
    image_h, image_w, ic = img.shape
    if image_h % tile_h or image_w % tile_w:
        raise ValueError("image dimensions must be exact multiples of tile size")
    padded = np.pad(img, ((halo, halo), (halo, halo), (0, 0)), mode="constant")
    patches = []
    for r in range(0, image_h, tile_h):
        for c in range(0, image_w, tile_w):
            patches.append(padded[r:r + tile_h + 2 * halo, c:c + tile_w + 2 * halo, :])
    return np.stack(patches, axis=0).astype(img.dtype, copy=False)


def iter_patch_chunks(patches: np.ndarray, chunk_size: int = 4):
    """Yield `(start, chunk)` views for row-major patch batches.

    The selected rn3pair production chunk is p4. Full-layer integration should
    call the fused ELF on chunks from this iterator, padding only the final chunk
    if the grid patch count is not divisible by 4.
    """
    a = np.asarray(patches)
    if a.ndim < 1:
        raise ValueError("patches must have a leading patch dimension")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, a.shape[0], chunk_size):
        yield start, a[start:start + chunk_size]


def pad_patch_chunk(chunk: np.ndarray, chunk_size: int = 4) -> tuple[np.ndarray, int]:
    """Pad a final short patch chunk to `chunk_size` by appending zeros.

    Returns `(padded_chunk, valid_count)`. The caller should discard padded
    outputs after NPU execution.
    """
    a = np.asarray(chunk)
    valid = a.shape[0]
    if valid > chunk_size:
        raise ValueError(f"chunk has {valid} patches, exceeds chunk_size={chunk_size}")
    if valid == chunk_size:
        return a, valid
    pad_shape = (chunk_size - valid,) + a.shape[1:]
    padded = np.concatenate([a, np.zeros(pad_shape, dtype=a.dtype)], axis=0)
    return padded, valid

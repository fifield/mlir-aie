"""CPU helpers for rn3 3x3+3x3 pair-fusion geometry tests."""
import numpy as np


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def extract_pair_patch(input_hwc: np.ndarray, tr: int, tc: int,
                       tile_h: int, tile_w: int) -> np.ndarray:
    """Extract the original-input patch needed by two stride-1 3x3 convs.

    For a final tile beginning at (tr*tile_h, tc*tile_w), two stacked 3x3
    convs need a two-pixel halo, i.e. a (tile+4)x(tile+4) patch. Values outside
    the image are zero padded.
    """
    h, w, c = input_hwc.shape
    out_r0 = tr * tile_h
    out_c0 = tc * tile_w
    r0 = out_r0 - 2
    c0 = out_c0 - 2
    patch = np.zeros((tile_h + 4, tile_w + 4, c), dtype=input_hwc.dtype)
    for pr in range(tile_h + 4):
        rr = r0 + pr
        if rr < 0 or rr >= h:
            continue
        for pc in range(tile_w + 4):
            cc = c0 + pc
            if 0 <= cc < w:
                patch[pr, pc, :] = input_hwc[rr, cc, :]
    return patch


def conv3x3_bn_silu_hwc(input_hwc: np.ndarray, weights_oihw: np.ndarray,
                        bn_w: np.ndarray, bn_b: np.ndarray,
                        padding: int = 1) -> np.ndarray:
    """Reference HWC/OIHW 3x3 conv + affine BN + SiLU."""
    h, w, ic = input_hwc.shape
    oc, ic_w, kh, kw = weights_oihw.shape
    assert ic == ic_w and kh == 3 and kw == 3
    padded = np.pad(input_hwc, ((padding, padding), (padding, padding), (0, 0)))
    out = np.zeros((h, w, oc), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            window = padded[r:r + 3, c:c + 3, :]
            for o in range(oc):
                # weights are O,I,H,W; window is H,W,I
                acc = np.sum(window * np.transpose(weights_oihw[o], (1, 2, 0)))
                out[r, c, o] = acc * bn_w[o] + bn_b[o]
    return silu(out).astype(np.float32)


def rn3_pair_tile_reference(input_hwc: np.ndarray, tr: int, tc: int,
                            tile_h: int, tile_w: int,
                            w1: np.ndarray, bn1_w: np.ndarray, bn1_b: np.ndarray,
                            w2: np.ndarray, bn2_w: np.ndarray, bn2_b: np.ndarray) -> np.ndarray:
    """Compute one fused rn3-pair output tile from a (tile+4) halo patch."""
    patch = extract_pair_patch(input_hwc, tr, tc, tile_h, tile_w)

    # Conv1 over the expanded patch with no additional external padding beyond
    # the already materialized two-pixel halo. Valid 3x3 produces (tile+2)^2.
    mid_h = tile_h + 2
    mid_w = tile_w + 2
    oc1 = w1.shape[0]
    mid = np.zeros((mid_h, mid_w, oc1), dtype=np.float32)
    for r in range(mid_h):
        for c in range(mid_w):
            window = patch[r:r + 3, c:c + 3, :]
            for o in range(oc1):
                acc = np.sum(window * np.transpose(w1[o], (1, 2, 0)))
                mid[r, c, o] = acc * bn1_w[o] + bn1_b[o]
    mid = silu(mid).astype(np.float32)

    # Conv2's padding is applied to the conv1 *output tensor*, not by extending
    # conv1 beyond the image. Intermediate positions outside the full-image
    # conv1 output domain must therefore be zero.
    h, w, _ = input_hwc.shape
    base_r = tr * tile_h - 1
    base_c = tc * tile_w - 1
    for r in range(mid_h):
        gr = base_r + r
        for c in range(mid_w):
            gc = base_c + c
            if gr < 0 or gr >= h or gc < 0 or gc >= w:
                mid[r, c, :] = 0.0

    # Conv2 valid over the (tile+2)x(tile+2) intermediate to produce tile^2.
    oc2 = w2.shape[0]
    out = np.zeros((tile_h, tile_w, oc2), dtype=np.float32)
    for r in range(tile_h):
        for c in range(tile_w):
            window = mid[r:r + 3, c:c + 3, :]
            for o in range(oc2):
                acc = np.sum(window * np.transpose(w2[o], (1, 2, 0)))
                out[r, c, o] = acc * bn2_w[o] + bn2_b[o]
    return silu(out).astype(np.float32)

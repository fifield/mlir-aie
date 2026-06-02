#!/usr/bin/env python3
"""CPU reference checks for fused rn3 3x3+3x3 tile semantics.

These tests prove the key geometry before the NPU kernel: a fused pair output
for one T×T tile must be computable from a (T+4)×(T+4) original-input patch.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rn3_pair_cpu import (
    extract_pair_patch,
    conv3x3_bn_silu_hwc,
    rn3_pair_tile_reference,
)


def _rng_inputs(h=16, w=16, c=4, oc=4):
    rng = np.random.default_rng(123)
    x = rng.normal(0, 0.4, size=(h, w, c)).astype(np.float32)
    w1 = rng.normal(0, 0.12, size=(oc, c, 3, 3)).astype(np.float32)
    b1w = rng.normal(1.0, 0.05, size=(oc,)).astype(np.float32)
    b1b = rng.normal(0.0, 0.05, size=(oc,)).astype(np.float32)
    w2 = rng.normal(0, 0.12, size=(oc, oc, 3, 3)).astype(np.float32)
    b2w = rng.normal(1.0, 0.05, size=(oc,)).astype(np.float32)
    b2b = rng.normal(0.0, 0.05, size=(oc,)).astype(np.float32)
    return x, w1, b1w, b1b, w2, b2w, b2b


def test_extract_pair_patch_has_two_conv_halo_at_image_corner():
    x = np.arange(4 * 4 * 1, dtype=np.float32).reshape(4, 4, 1)
    patch = extract_pair_patch(x, tr=0, tc=0, tile_h=2, tile_w=2)
    assert patch.shape == (6, 6, 1)
    assert np.all(patch[:2, :, :] == 0)
    assert np.all(patch[:, :2, :] == 0)
    np.testing.assert_array_equal(patch[2:6, 2:6, 0], x[:, :, 0])


def test_rn3_pair_tile_matches_full_two_conv_reference_for_interior_tile():
    x, w1, b1w, b1b, w2, b2w, b2b = _rng_inputs(h=24, w=24, c=4, oc=4)
    y1 = conv3x3_bn_silu_hwc(x, w1, b1w, b1b, padding=1)
    y2 = conv3x3_bn_silu_hwc(y1, w2, b2w, b2b, padding=1)

    tile_h = tile_w = 8
    tr = tc = 1
    tile = rn3_pair_tile_reference(x, tr, tc, tile_h, tile_w, w1, b1w, b1b, w2, b2w, b2b)
    expected = y2[tr * tile_h:(tr + 1) * tile_h, tc * tile_w:(tc + 1) * tile_w, :]
    np.testing.assert_allclose(tile, expected, rtol=1e-5, atol=1e-5)


def test_rn3_pair_tile_matches_full_two_conv_reference_at_padded_corner():
    x, w1, b1w, b1b, w2, b2w, b2b = _rng_inputs(h=16, w=16, c=4, oc=4)
    y1 = conv3x3_bn_silu_hwc(x, w1, b1w, b1b, padding=1)
    y2 = conv3x3_bn_silu_hwc(y1, w2, b2w, b2b, padding=1)

    tile = rn3_pair_tile_reference(x, 0, 0, 8, 8, w1, b1w, b1b, w2, b2w, b2b)
    np.testing.assert_allclose(tile, y2[:8, :8, :], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_extract_pair_patch_has_two_conv_halo_at_image_corner()
    test_rn3_pair_tile_matches_full_two_conv_reference_for_interior_tile()
    test_rn3_pair_tile_matches_full_two_conv_reference_at_padded_corner()
    print("PASS: rn3 pair CPU tile-reference tests")

#!/usr/bin/env python3
import numpy as np

from rn3_pair_layout import (
    block_major_to_hwc,
    hwc_to_block_major,
    iter_patch_chunks,
    pack_rn3_pair_input_patches,
    pad_patch_chunk,
    scatter_tile_hwc_to_image,
)


def test_block_major_hwc_roundtrip():
    rng = np.random.default_rng(1)
    block = rng.normal(size=(4, 12, 8, 8, 4)).astype(np.float32)
    hwc = block_major_to_hwc(block)
    assert hwc.shape == (4, 8, 8, 48)
    np.testing.assert_array_equal(hwc_to_block_major(hwc, 4), block)


def test_block_major_channel_order():
    block = np.zeros((1, 3, 2, 2, 4), dtype=np.int32)
    for b in range(3):
        for o in range(4):
            block[0, b, :, :, o] = b * 4 + o
    hwc = block_major_to_hwc(block)
    expected_channels = np.arange(12, dtype=np.int32)
    np.testing.assert_array_equal(hwc[0, 0, 0, :], expected_channels)


def test_pack_and_scatter_row_major_tiles():
    img = np.arange(16 * 16 * 2, dtype=np.float32).reshape(16, 16, 2)
    patches = pack_rn3_pair_input_patches(img, 8, 8, halo=2)
    assert patches.shape == (4, 12, 12, 2)
    # First patch has zero halo in top-left corner and original image at halo offset.
    assert patches[0, 0, 0, 0] == 0
    np.testing.assert_array_equal(patches[0, 2:10, 2:10, :], img[:8, :8, :])
    np.testing.assert_array_equal(patches[1, 2:10, 2:10, :], img[:8, 8:16, :])

    tiles = np.arange(4 * 8 * 8 * 3, dtype=np.float32).reshape(4, 8, 8, 3)
    scattered = scatter_tile_hwc_to_image(tiles, 16, 16, 8, 8)
    np.testing.assert_array_equal(scattered[:8, :8, :], tiles[0])
    np.testing.assert_array_equal(scattered[:8, 8:16, :], tiles[1])
    np.testing.assert_array_equal(scattered[8:16, :8, :], tiles[2])
    np.testing.assert_array_equal(scattered[8:16, 8:16, :], tiles[3])


def test_p4_chunking_with_final_padding():
    patches = np.arange(10 * 12 * 12 * 3, dtype=np.float32).reshape(10, 12, 12, 3)
    chunks = list(iter_patch_chunks(patches, 4))
    assert [s for s, _ in chunks] == [0, 4, 8]
    assert [c.shape[0] for _, c in chunks] == [4, 4, 2]
    padded, valid = pad_patch_chunk(chunks[-1][1], 4)
    assert valid == 2
    assert padded.shape == (4, 12, 12, 3)
    np.testing.assert_array_equal(padded[:2], patches[8:10])
    np.testing.assert_array_equal(padded[2:], np.zeros((2, 12, 12, 3), dtype=np.float32))


if __name__ == "__main__":
    test_block_major_hwc_roundtrip()
    test_block_major_channel_order()
    test_pack_and_scatter_row_major_tiles()
    test_p4_chunking_with_final_padding()
    print("PASS: rn3 pair layout helpers")

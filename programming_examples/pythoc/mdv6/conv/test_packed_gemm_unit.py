#!/usr/bin/env python3
"""Pure-Python checks for packed GEMM spatial fanout helpers.

These deliberately avoid pyxrt/NPU execution. The NPU correctness test should
come after the packed ELF builds; these guard the ABI/layout decisions first.
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build_packed_gemm import packed_gemm_elf_name, packed_host_sizes
from run_tiled_mc import _pack_gemm_spatial_inputs


class PackedGemmUnitTests(unittest.TestCase):
    def test_packed_gemm_elf_name_includes_batch_count_and_kblock_when_present(self):
        self.assertEqual(
            packed_gemm_elf_name(tile_m=164, ic=96, oc=48, k_block=0, ppc=1, n_batches=10),
            "merged_gemm_t164_ic96_oc48_p1_x10_packed",
        )
        self.assertEqual(
            packed_gemm_elf_name(tile_m=64, ic=256, oc=128, k_block=64, ppc=1, n_batches=4),
            "merged_gemm_t64_ic256_oc128_kb64_p1_x4_packed",
        )

    def test_packed_host_sizes_are_exact_concatenation_of_old_x1_batch_layout(self):
        self.assertEqual(
            packed_host_sizes(n_batches=3, host_in_size=1024, host_out_size=2048),
            (3072, 6144),
        )

    def test_pack_gemm_spatial_inputs_concatenates_old_per_batch_buffers(self):
        # 2 cores, 2 slots/core, tile_m=2, IC=3 => input_size=6, total_slots=4.
        # pixels_per_call = 8 pixels. M=10 requires two batches; batch1 has
        # only 2 active pixels and should pad trailing slots with slot0.
        input_flat = np.arange(10 * 3, dtype=np.uint16).reshape(10, 3)
        packed, n_batches = _pack_gemm_spatial_inputs(
            input_flat,
            total_slots=4,
            input_size=6,
            tile_m=2,
            pixels_per_call=8,
        )
        self.assertEqual(n_batches, 2)
        self.assertEqual(packed.shape, (2 * 4 * 6,))

        batch0 = packed[:24].reshape(4, 6)
        np.testing.assert_array_equal(batch0[0], input_flat[0:2].reshape(-1))
        np.testing.assert_array_equal(batch0[1], input_flat[2:4].reshape(-1))
        np.testing.assert_array_equal(batch0[2], input_flat[4:6].reshape(-1))
        np.testing.assert_array_equal(batch0[3], input_flat[6:8].reshape(-1))

        batch1 = packed[24:].reshape(4, 6)
        slot0 = input_flat[8:10].reshape(-1)
        np.testing.assert_array_equal(batch1[0], slot0)
        np.testing.assert_array_equal(batch1[1], slot0)
        np.testing.assert_array_equal(batch1[2], slot0)
        np.testing.assert_array_equal(batch1[3], slot0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

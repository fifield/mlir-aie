"""CPU-only routing checks: python -m unittest test_regime_routes."""
import unittest

from regime_config import (
    conv_regime_for_layer, gemm_regime_for_layer, regime_artifacts,
    regime_route_inventory,
)


class RegimeRoutesTest(unittest.TestCase):
    def test_legacy_selection_preserved(self):
        self.assertEqual(conv_regime_for_layer("mc_re4_c3").xclbin_name,
                         "regime_shared_conv3x3")
        artifact, member = gemm_regime_for_layer("gemm_re6_rn1", 96, 48, 0)
        self.assertEqual(artifact.xclbin_name, "regime_r1_gemm_non_k")
        self.assertEqual(member.tile_m, 44)
        self.assertIsNotNone(conv_regime_for_layer("mc_ftconv0"))

    def test_subset_restores_spatial_tiling_and_excludes_stride2(self):
        route = "per-regime-r1-r3"
        artifact = conv_regime_for_layer("mc_re4_c3", route=route)
        self.assertEqual((artifact.xclbin_name, artifact.tile_h),
                         ("regime_r1_conv3x3", 8))
        self.assertIsNone(conv_regime_for_layer("mc_ftconv0", route=route))
        artifact, member = gemm_regime_for_layer("gemm_re6_rn1", 96, 48, 0, route)
        self.assertEqual((artifact.xclbin_name, member.tile_m),
                         ("regime_r2_gemm_non_k", 100))

    def test_overloaded_runtime_names_select_by_shape(self):
        route = "per-regime-r1-r3"
        for ic, oc, kb, name in (
            (128, 128, 0, "regime_r3_gemm_non_k"),
            (256, 256, 64, "regime_r3_gemm_kblocked"),
            (256, 128, 128, "regime_r3_gemm_kblocked"),
        ):
            artifact, member = gemm_regime_for_layer("gemm_re8_c1", ic, oc, kb, route)
            self.assertEqual(artifact.xclbin_name, name)
            self.assertEqual((member.ic, member.oc, member.k_block), (ic, oc, kb))
        self.assertEqual(gemm_regime_for_layer("gemm_re8_c1", 999, 128, 0, route),
                         (None, None))

    def test_inventory_distinguishes_artifacts_from_instruction_handles(self):
        rows = regime_route_inventory("per-regime-r1-r3")
        self.assertEqual(len({r["xclbin_name"] for r in rows}), 9)
        self.assertEqual(len({(r["xclbin_name"], r["insts_name"]) for r in rows}), 26)
        for row in rows:
            self.assertNotIn("shared", row["xclbin_name"])
            self.assertNotIn("r5", row["xclbin_name"])
        self.assertEqual(len(regime_route_inventory("per-regime-r1-r3", False)), 16)

    def test_unknown_route_fails(self):
        with self.assertRaises(ValueError):
            regime_artifacts("r1-r3-typo")


if __name__ == "__main__":
    unittest.main()

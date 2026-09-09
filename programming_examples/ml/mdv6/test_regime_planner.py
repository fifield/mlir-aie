"""Device-free regression and boundary checks for the MDV6 screening model."""
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

import regime_planner as planner


class PlannerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = planner.build_report()
        cls.plans = {p["name"]: p for p in cls.report["plans"]}

    def test_full_graph_matches_independent_recorded_launch_counts(self):
        # Measurements in PERF_PLAN.md. These are NOT inputs to the model.
        self.assertEqual(self.plans["standalone"]["total_host_launches"], 453)
        self.assertEqual(self.plans["standalone"]["resident_xclbins_est"], 32)
        self.assertEqual(self.plans["plus_r5"]["total_host_launches"], 742)
        self.assertEqual(self.plans["plus_shared_k"]["total_host_launches"], 757)
        self.assertEqual(self.plans["all_shared"]["total_host_launches"], 933)

    def test_baseline_calibration_and_held_out_regression_direction(self):
        baseline = json.loads((planner.ROOT / "profile_baseline.json").read_text())
        self.assertAlmostEqual(self.plans["standalone"]["estimated_wall_ms"], baseline["wall_ms"])
        for before, after in (("standalone", "plus_r5"), ("plus_r5", "plus_shared_k"), ("plus_shared_k", "all_shared")):
            self.assertGreater(self.plans[after]["estimated_wall_ms"], self.plans[before]["estimated_wall_ms"])

    def test_per_regime_launch_delta_has_checked_layer_explanation(self):
        check = self.report["per_regime_launch_check"]
        self.assertEqual(check["predicted_calls"], 466)
        self.assertEqual(sum(r["delta"] for r in check["layer_deltas"]), 13)
        self.assertEqual(sum(max(0, r["delta"]) for r in check["layer_deltas"]), 15)
        self.assertEqual(sum(min(0, r["delta"]) for r in check["layer_deltas"]), -2)
        self.assertEqual(self.plans["per_regime_r1_r3"]["total_host_launches"], check["predicted_calls"])

    def test_stem_retile_explains_launch_growth(self):
        rows = {r["candidate_regime"]: r for r in self.report["candidates"] if r["runtime_name"] == "mc_ftconv0"}
        self.assertEqual(rows["mc_ftconv0"]["calls_today"], 8)
        self.assertEqual(rows["regime_r5_stride2_conv3x3"]["calls_today"], 200)
        self.assertIn("launch count grows >25%", rows["regime_r5_stride2_conv3x3"]["performance_flags"])

    def test_rtp_envelope_padding_is_not_always_compute_padding(self):
        layer = planner.Layer("tiny", "mc_tiny", 8, 8, 8, 8, "conv", 8, 8, 8)
        small = planner.Envelope("small", 8, 8, 8, 8, 1)
        large = planner.Envelope("large", 8, 8, 64, 32, 1, active_oc=8)
        a, b = planner.measure(layer, small), planner.measure(layer, large)
        self.assertEqual(a["padded_macs"], b["padded_macs"])
        self.assertGreater(b["transfer_bytes"], a["transfer_bytes"])

    def test_kblocked_channels_are_executed_padding(self):
        layer = planner.Layer("g", "gemm_g", 4, 4, 64, 64, "gemm")
        small = planner.Envelope("small", 4, 1, 64, 64, 1, 32)
        large = planner.Envelope("large", 4, 1, 128, 128, 1, 32)
        self.assertEqual(planner.measure(layer, large)["padded_macs"], 4 * planner.measure(layer, small)["padded_macs"])

    def test_underoccupied_cores_and_fit_rejection(self):
        layer = planner.Layer("g", "gemm_g", 4, 4, 64, 64, "gemm")
        env = planner.Envelope("e", 4, 1, 64, 64, 1)
        small, full = planner.measure(layer, env, 4), planner.measure(layer, env, 32)
        self.assertEqual(small["calls_today"], full["calls_today"])
        self.assertEqual(small["core_occupancy"], 1)
        self.assertEqual(full["core_occupancy"], 0.125)
        oversized = planner.Envelope("bad", 256, 1, 512, 512, 32)
        rejected = planner.measure(layer, oversized)
        self.assertFalse(rejected["fits_l1"])
        self.assertFalse(rejected["fits_l2"])
        self.assertFalse(rejected["feasible_estimate"])

    def test_fifo_memory_and_kblocked_weight_replay_match_generator(self):
        # Four tiles fill a 4-core call with PPC1. PPC4 runs the same one
        # call, streaming/replaying four slots, so differences isolate PPC.
        layer = planner.Layer("g", "gemm_g", 4, 4, 64, 64, "gemm")
        env = planner.Envelope("g", 4, 1, 64, 64, 1, 32)
        one = planner.measure(layer, env, 4)
        four = planner.measure(layer, replace(env, ppc=4), 4)
        self.assertEqual(one["l1_bytes_est"], four["l1_bytes_est"])
        self.assertEqual(one["l2_bytes_per_column_est"], four["l2_bytes_per_column_est"])
        self.assertEqual(four["weight_bytes"], 4 * one["weight_bytes"])
        # Buffer sizes: input 4*64*2, output 4*64*2, wt=(32*64+128)*2.
        self.assertEqual(one["l1_bytes_est"], 512 + 512 + 4352 + 8192 + 32)
        self.assertEqual(one["l2_bytes_per_column_est"], 4 * (512 + 512) + 4352)
        conv = planner.Layer("c", "mc_c", 8, 8, 8, 8, "conv", 4, 4, 8)
        cenv = planner.Envelope("c", 4, 4, 8, 8, 1)
        c1 = planner.measure(conv, cenv, 4)
        c4 = planner.measure(conv, replace(cenv, ppc=4), 4)
        self.assertGreater(c4["l2_bytes_per_column_est"], c1["l2_bytes_per_column_est"])
        self.assertEqual(c1["weight_bytes"], c4["weight_bytes"])

    def test_all_existing_candidate_efficiencies_are_physical(self):
        helpers = planner.gemm_helpers()
        for layer in planner.model_layers():
            for env in planner.candidates(layer, helpers):
                for cores in (4, 8, 16, 24, 32):
                    row = planner.measure(layer, env, cores)
                    self.assertGreater(row["compute_efficiency"], 0)
                    self.assertLessEqual(row["compute_efficiency"], 1)

    def test_no_feasible_candidate_reports_layer_and_reasons(self):
        layer = planner.Layer("oversized_layer", "gemm_bad", 4, 4, 512, 512, "gemm")
        env = planner.Envelope("oversized_env", 256, 1, 512, 512, 32)
        with self.assertRaisesRegex(ValueError, "No feasible candidate for layer oversized_layer.*L1"):
            planner.best_existing_candidate(layer, [env], lambda row: row["padded_macs"])
        with self.assertRaisesRegex(ValueError, "No feasible candidate for layer oversized_layer"):
            planner.best_existing_candidate(layer, [], lambda row: row["padded_macs"])

    def test_baseline_drift_fails_closed(self):
        baseline = json.loads((planner.ROOT / "profile_baseline.json").read_text())
        baseline["n_launches"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(json.dumps(baseline))
            with self.assertRaisesRegex(ValueError, "recalibrate explicitly"):
                planner.build_report(path)


if __name__ == "__main__":
    unittest.main()

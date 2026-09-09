import unittest

from sppelan import fusion_schedule as s


class ScheduleTests(unittest.TestCase):
    def test_budgets(self):
        report = s.validate()
        self.assertEqual(report["l1_peak_bytes_per_worker"], 58432)
        self.assertEqual(report["l2_peak_bytes_per_column"], 221440)
        self.assertEqual(report["resident_feature_bytes"], 409600)

    def test_external_contracts(self):
        for contract in s.boundaries():
            self.assertEqual(contract.nbytes, 204800)
            self.assertEqual(contract.storage, "external_bo")
            self.assertEqual(contract.layout, "HWC")

    def test_reject_overbudget_or_invalid_records(self):
        for storage, limit in (("L1", s.L1_LIMIT), ("L2", s.L2_LIMIT)):
            with self.subTest(storage=storage), self.assertRaisesRegex(ValueError, "peak exceeds"):
                s.validate(s.reservations() + (s.Reservation("extra", storage, limit, 0, 1),))
        for record in (s.Reservation("bad", "L3", 1, 0, 1),
                       s.Reservation("bad", "L1", 0, 0, 1),
                       s.Reservation("bad", "L1", 1, 1, 0)):
            with self.subTest(record=record), self.assertRaisesRegex(ValueError, "invalid storage"):
                s.validate((record,))
        with self.assertRaisesRegex(ValueError, "reservations are required"):
            s.validate(())

    def test_gather_complete_and_ordered(self):
        # Every destination element is written exactly once; decode each source
        # offset and prove that its pool level/pixel/channel is the desired K.
        for stripe in range(25):
            destinations = set()
            for segment in s.gather_segments(stripe):
                for p in range(segment["rows"]):
                    for c in range(segment["width"]):
                        src = segment["source_offset"] + p * segment["source_stride"] + c
                        row_level, pixel_channel = divmod(src, s.H * s.W * s.SHARD)
                        row, level = divmod(row_level, 4)
                        pixel, local_c = divmod(pixel_channel, s.SHARD)
                        k = level * s.C_NECK + (segment["source_column"] * s.ROWS + row) * s.SHARD + local_c
                        dst = segment["destination_offset"] + p * segment["destination_stride"] + c
                        self.assertEqual(pixel, stripe * s.STRIPE + p)
                        self.assertEqual(dst, p * 512 + k)
                        self.assertNotIn(dst, destinations)
                        destinations.add(dst)
            self.assertEqual(destinations, set(range(16 * 512)))

    def test_phase_reuse_not_cumulative_allocation(self):
        all_l1 = sum(r.nbytes for r in s.reservations() if r.storage == "L1")
        self.assertGreater(all_l1, s.L1_LIMIT)
        self.assertLess(s.peak_bytes("L1"), s.L1_LIMIT)

    def test_reject_invalid_indices(self):
        for k in (-1, 512):
            with self.assertRaises(ValueError):
                s.source_for_channel(k)
        for stripe in (-1, 25):
            with self.assertRaises(ValueError):
                list(s.gather_segments(stripe))


if __name__ == "__main__":
    unittest.main()

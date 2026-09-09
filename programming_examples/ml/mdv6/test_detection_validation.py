"""CPU-only checks for the stream's numerical failure gates."""
import unittest
import torch
from detection_validation import compare_detection_outputs


class DetectionValidationTest(unittest.TestCase):
    def setUp(self):
        self.reference = [(torch.ones(2), torch.ones(2), torch.ones(2)) for _ in range(3)]
        self.actual = [tuple(t.clone() for t in scale) for scale in self.reference]

    def compare(self):
        return compare_detection_outputs(self.reference, self.actual, 0.5, 0.1)

    def test_matching_outputs_pass(self):
        self.assertEqual(self.compare(), dict(finite=True, max_class_diff=0.0,
                                             max_vector_diff=0.0, ok=True))

    def test_later_scale_nan_and_anchor_inf_fail(self):
        for scale, tensor, value in ((2, 0, float('nan')), (1, 1, float('inf'))):
            with self.subTest(scale=scale, tensor=tensor):
                self.setUp()
                self.actual[scale][tensor][0] = value
                self.assertFalse(self.compare()['ok'])
                self.assertIsNone(self.compare()['max_class_diff'])

    def test_reference_nonfinite_fails(self):
        self.reference[2][2][0] = float('nan')
        self.assertFalse(self.compare()['ok'])

    def test_threshold_exceeded_fails(self):
        self.actual[2][2][0] += 0.2
        self.assertFalse(self.compare()['ok'])

    def test_missing_scale_fails(self):
        self.actual.pop()
        with self.assertRaises(ValueError):
            self.compare()

    def test_broadcastable_shape_mismatch_fails(self):
        self.actual[0] = (torch.ones(1), torch.ones(2), torch.ones(2))
        with self.assertRaises(ValueError):
            self.compare()


if __name__ == '__main__':
    unittest.main()

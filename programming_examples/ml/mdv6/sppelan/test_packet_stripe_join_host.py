"""CPU address coverage, stripe/sender identity and fail-stop tests."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from sppelan.packet_stripe_join_host import PacketStripeJoin, INPUT_SHAPE, OUTPUT_SHAPE, ELEMENTS, frame_input, oracle, validate_output
from sppelan.test_packet_stripe_join import validate_counts, save_failure


class StripeHostTests(unittest.TestCase):
    def test_every_source_destination_coordinate_once(self):
        stripe, worker, pixel, channel = np.indices(INPUT_SHAPE)
        src = stripe * 1024 + worker * 256 + pixel * 16 + channel
        dst = stripe * 1024 + pixel * 64 + worker * 16 + channel
        np.testing.assert_array_equal(np.sort(src.reshape(-1)), np.arange(ELEMENTS))
        np.testing.assert_array_equal(np.sort(dst.reshape(-1)), np.arange(ELEMENTS))
        bits = frame_input(0)[1]
        expected = np.empty(ELEMENTS, np.uint16)
        expected[dst.reshape(-1)] = bits.reshape(-1)[src.reshape(-1)] ^ (0x1111 * (worker.reshape(-1) + 1)).astype(np.uint16)
        np.testing.assert_array_equal(oracle(bits).reshape(-1), expected)
        for s, w, p, c in ((0, 0, 0, 0), (0, 3, 15, 15), (1, 0, 0, 0), (12, 2, 7, 9), (24, 3, 15, 15)):
            self.assertEqual(oracle(bits)[s, p, w * 16 + c], bits[s, w, p, c] ^ (0x1111 * (w + 1)))

    def test_reject_sender_reordering_and_reused_stripes(self):
        for frame in (0, 1):
            expected = oracle(frame_input(frame)[1])
            worker_swapped = expected.reshape(25, 16, 4, 16)[:, :, [1, 0, 2, 3], :].reshape(OUTPUT_SHAPE)
            stale = np.broadcast_to(expected[0], OUTPUT_SHAPE).copy()
            for wrong in (worker_swapped, stale, np.roll(expected, 1, axis=0)):
                with self.assertRaisesRegex(RuntimeError, 'stripe/pixel/channel'):
                    validate_output(wrong, expected)
        bits = frame_input(0)[1]
        with self.assertRaises(RuntimeError):
            validate_output(bits.transpose(0, 2, 1, 3).reshape(OUTPUT_SHAPE), oracle(bits))

    def test_patterns_own_oracle_and_changing_random(self):
        for frame in range(6):
            bits = frame_input(frame)[1]
            before = bits.copy()
            result = oracle(bits)
            validate_output(result, result.copy())
            self.assertFalse(np.shares_memory(result, bits))
            np.testing.assert_array_equal(bits, before)
        self.assertFalse(np.array_equal(frame_input(2)[1], frame_input(8)[1]))
        self.assertIn(0x7fc1, frame_input(5)[1])

    def test_bad_contracts(self):
        for bits in (np.zeros(ELEMENTS, np.uint16), np.zeros(INPUT_SHAPE, np.float32)):
            with self.assertRaises(ValueError):
                oracle(bits)
        for output in (np.zeros(ELEMENTS, np.uint16), np.zeros(OUTPUT_SHAPE, np.float32)):
            with self.assertRaises(RuntimeError):
                validate_output(output, np.zeros(OUTPUT_SHAPE, np.uint16))
        with self.assertRaises(ValueError):
            frame_input(-1)

    def make_probe(self):
        backend = Mock()
        def buffer(n, dtype):
            result = Mock(data=np.zeros(n, dtype))
            result.numpy.return_value = result.data
            return result
        backend.iron.zeros.side_effect = buffer
        backend.DefaultNPURuntime.run.return_value.is_success.return_value = True
        with tempfile.TemporaryDirectory() as directory:
            for name in ('packet_stripe_join.xclbin', 'packet_stripe_join.bin'):
                Path(directory, name).touch()
            probe = PacketStripeJoin(directory, backend)
        return probe, backend

    def test_one_submission_per_whole_frame_persistent_buffers(self):
        probe, backend = self.make_probe()
        for frame in range(2):
            bits = frame_input(frame)[1]
            before = bits.copy()
            output = probe.run(bits)
            np.testing.assert_array_equal(bits, before)
            np.testing.assert_array_equal(probe.input.data, bits.reshape(-1))
            self.assertFalse(np.shares_memory(output, probe.output.data))
        backend.DefaultNPURuntime.load.assert_called_once()
        self.assertEqual(backend.iron.zeros.call_count, 2)
        self.assertEqual(backend.DefaultNPURuntime.run.call_count, 2)
        self.assertEqual(probe.input._sync_to_device.call_count, 2)
        self.assertEqual(probe.output.numpy.call_count, 2)
        backend.DefaultNPURuntime.run.assert_called_with(probe.handle, [probe.input, probe.output])

    def test_failures_poison_without_retry(self):
        for stage in ('upload', 'run', 'result', 'readback', 'output_contract'):
            probe, backend = self.make_probe()
            if stage == 'upload':
                probe.input._sync_to_device.side_effect = RuntimeError('upload failed')
            elif stage == 'run':
                backend.DefaultNPURuntime.run.side_effect = RuntimeError('run failed')
            elif stage == 'result':
                backend.DefaultNPURuntime.run.return_value.is_success.return_value = False
            elif stage == 'readback':
                probe.output.numpy.side_effect = RuntimeError('readback failed')
            else:
                probe.output.numpy.return_value = np.zeros(1, np.uint16)
            with self.assertRaises(RuntimeError):
                probe.run(frame_input(0)[1])
            with self.assertRaisesRegex(RuntimeError, 'invalid after failure'):
                probe.run(frame_input(1)[1])
            self.assertEqual(backend.DefaultNPURuntime.run.call_count, 0 if stage == 'upload' else 1)
            backend.DefaultNPURuntime.load.assert_called_once()

    def test_missing_artifacts_never_load(self):
        backend = Mock()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                PacketStripeJoin(directory, backend)
        backend.DefaultNPURuntime.load.assert_not_called()

    def test_counts_and_exclusive_evidence(self):
        counts = dict(run_calls=1, returned_runs=1, completed_runs=1, load_calls=0,
                      sync_to_calls=1, sync_from_calls=1, sync_to_bytes=51200, sync_from_bytes=51200)
        validate_counts(counts)
        for key in counts:
            bad = dict(counts)
            bad[key] += 1
            with self.assertRaises(RuntimeError):
                validate_counts(bad)
        with tempfile.TemporaryDirectory() as directory:
            bits = frame_input(0)[1]
            expected = oracle(bits)
            path = save_failure(directory, 0, 'index', bits, expected, None, 'failure')
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved['input'], bits)
                np.testing.assert_array_equal(saved['expected'], expected)
            with self.assertRaises(FileExistsError):
                save_failure(directory, 0, 'index', bits, expected, expected, 'later')


if __name__ == '__main__':
    unittest.main()

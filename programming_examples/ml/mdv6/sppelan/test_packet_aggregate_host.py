"""CPU contract, sender-order and fail-stop tests for packet aggregation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from sppelan.packet_aggregate_host import PacketAggregate, SHAPE, ELEMENTS, frame_input, oracle, validate_output
from sppelan.test_packet_aggregate import validate_counts, save_failure


class PacketHostTests(unittest.TestCase):
    def test_unique_indices_and_worker_transform(self):
        bits = frame_input(0)[1]
        self.assertEqual(np.unique(bits).size, ELEMENTS)
        before = bits.copy()
        expected = oracle(bits)
        for worker in range(4):
            np.testing.assert_array_equal(expected[worker], bits[worker] ^ (0x1111 * (worker + 1)))
        np.testing.assert_array_equal(bits, before)
        with self.assertRaises(RuntimeError):
            validate_output(bits, expected)  # Bypassing workers cannot pass.
        with self.assertRaises(RuntimeError):
            validate_output(bits ^ 0x1111, expected)  # One worker cannot impersonate all four.

    def test_reject_reordered_senders_levels_and_lanes(self):
        for frame in (0, 1):
            expected = oracle(frame_input(frame)[1])
            with self.assertRaisesRegex(RuntimeError, 'worker/level/pixel/lane'):
                validate_output(expected[[1, 0, 2, 3]], expected)
        expected = oracle(frame_input(0)[1])
        for wrong in (expected[:, ::-1].copy(), expected[:, :, :, ::-1].copy()):
            with self.assertRaises(RuntimeError):
                validate_output(wrong, expected)

    def test_all_patterns_and_owned_output(self):
        for frame in range(6):
            _, bits = frame_input(frame)
            expected = oracle(bits)
            validate_output(expected, expected.copy())
            self.assertFalse(np.shares_memory(expected, bits))
        self.assertFalse(np.array_equal(frame_input(1)[1], frame_input(5)[1]))
        self.assertFalse(np.array_equal(frame_input(1)[1], frame_input(7)[1]))
        self.assertIn(0x7fc1, frame_input(4)[1])

    def test_bad_contracts(self):
        for bits in (np.zeros(ELEMENTS, np.uint16), np.zeros(SHAPE, np.float32)):
            with self.assertRaises(ValueError):
                oracle(bits)
            with self.assertRaises(RuntimeError):
                validate_output(bits, np.zeros(SHAPE, np.uint16))
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
            for name in ('packet_aggregate.xclbin', 'packet_aggregate.bin'):
                Path(directory, name).touch()
            probe = PacketAggregate(directory, backend)
        return probe, backend

    def test_rebinding_persistent_arenas(self):
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
                PacketAggregate(directory, backend)
        backend.DefaultNPURuntime.load.assert_not_called()

    def test_counts_and_exclusive_evidence(self):
        counts = dict(run_calls=1, returned_runs=1, completed_runs=1, load_calls=0,
                      sync_to_calls=1, sync_from_calls=1, sync_to_bytes=102400, sync_from_bytes=102400)
        validate_counts(counts)
        for key in counts:
            bad = dict(counts)
            bad[key] += 1
            with self.assertRaises(RuntimeError):
                validate_counts(bad)
        with tempfile.TemporaryDirectory() as directory:
            bits = frame_input(0)[1]
            path = save_failure(directory, 0, 'index', bits, bits, None, 'failure')
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved['input'], bits)
            with self.assertRaises(FileExistsError):
                save_failure(directory, 0, 'index', bits, bits, bits, 'later')


if __name__ == '__main__':
    unittest.main()

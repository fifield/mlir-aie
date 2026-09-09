"""CPU oracle independence, all-axis coverage and fail-stop host tests."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from sppelan.packet_gather_host import PacketGather, INPUT_SHAPE, OUTPUT_SHAPE, INPUT_ELEMENTS, OUTPUT_ELEMENTS, frame_input, oracle, validate_output
from sppelan.test_packet_gather import validate_counts, save_failure


class PacketGatherHostTests(unittest.TestCase):
    def test_coordinate_oracle_matches_independent_transpose(self):
        tags = (np.arange(1, 5, dtype=np.uint16) * 0x1111).reshape(1, 4, 1, 1, 1)
        for frame in range(6):
            bits = frame_input(frame)[1]
            independent = (bits ^ tags).transpose(3, 2, 0, 1, 4).reshape(25, 16, 512)
            expected = oracle(bits)
            for destination in range(4):
                np.testing.assert_array_equal(expected[destination], independent)
            self.assertFalse(np.shares_memory(expected, bits))

    def test_paired_sentinels_cover_all_source_axes_and_destinations(self):
        low, high = frame_input(0)[1], frame_input(1)[1]
        self.assertLess(np.unique(low).size, INPUT_ELEMENTS)
        self.assertLess(np.unique(high).size, INPUT_ELEMENTS)
        np.testing.assert_array_equal((low.astype(np.uint32) | high.astype(np.uint32) << 16).reshape(-1),
                                      np.arange(INPUT_ELEMENTS))
        out_low, out_high = oracle(low), oracle(high)
        pixel, level, source, worker, lane = np.indices((400, 4, 4, 4, 8))
        ids = ((((source * 4 + worker) * 4 + level) * 400 + pixel) * 8 + lane).reshape(25, 16, 512)
        self.assertEqual(np.unique(ids).size, INPUT_ELEMENTS)
        tags = (0x1111 * (worker + 1)).astype(np.uint16).reshape(25, 16, 512)
        for destination in range(4):
            recovered = (out_low[destination] ^ tags).astype(np.uint32) | ((out_high[destination] ^ tags).astype(np.uint32) << 16)
            np.testing.assert_array_equal(recovered, ids)

    def test_reject_wrong_source_level_worker_stripe_and_missing_destination(self):
        bits = frame_input(2)[1]
        expected = oracle(bits)
        mutations = [oracle(bits[[1, 0, 2, 3]]), oracle(bits[:, [1, 0, 2, 3]]),
                     oracle(bits[:, :, [1, 0, 2, 3]]), oracle(bits.transpose(0, 2, 1, 3, 4)),
                     np.roll(expected, 1, axis=1)]
        missing = expected.copy()
        missing[3] = 0
        mutations.append(missing)
        stale = expected.copy()
        stale[3] = oracle(frame_input(8)[1])[3]
        mutations.append(stale)
        for wrong in mutations:
            with self.assertRaisesRegex(RuntimeError, 'destination/stripe/pixel/channel'):
                validate_output(wrong, expected)
        bypass = np.broadcast_to(bits.transpose(3, 2, 0, 1, 4).reshape(25, 16, 512), OUTPUT_SHAPE).copy()
        with self.assertRaises(RuntimeError):
            validate_output(bypass, expected)

    def test_patterns_change_and_inputs_remain_unmodified(self):
        for frame in range(6):
            bits = frame_input(frame)[1]
            before = bits.copy()
            oracle(bits)
            np.testing.assert_array_equal(bits, before)
        self.assertFalse(np.array_equal(frame_input(2)[1], frame_input(8)[1]))
        self.assertIn(0x7fc1, frame_input(5)[1])

    def test_bad_contracts(self):
        for bits in (np.zeros(INPUT_ELEMENTS, np.uint16), np.zeros(INPUT_SHAPE, np.float32)):
            with self.assertRaises(ValueError):
                oracle(bits)
        for output in (np.zeros(OUTPUT_ELEMENTS, np.uint16), np.zeros(OUTPUT_SHAPE, np.float32)):
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
            for name in ('packet_gather.xclbin', 'packet_gather.bin'):
                Path(directory, name).touch()
            probe = PacketGather(directory, backend)
        return probe, backend

    def test_native_upload_and_persistent_no_intermediate_host_buffers(self):
        probe, backend = self.make_probe()
        for frame in range(2):
            bits = frame_input(frame)[1]
            before = bits.copy()
            out = probe.run(bits)
            np.testing.assert_array_equal(probe.input.data, bits.reshape(-1))
            np.testing.assert_array_equal(bits, before)
            self.assertFalse(np.shares_memory(out, probe.output.data))
        self.assertEqual(backend.iron.zeros.call_count, 2)
        backend.DefaultNPURuntime.load.assert_called_once()
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
                PacketGather(directory, backend)
        backend.DefaultNPURuntime.load.assert_not_called()

    def test_counts_and_exclusive_evidence(self):
        counts = dict(run_calls=1, returned_runs=1, completed_runs=1, load_calls=0,
                      sync_to_calls=1, sync_from_calls=1, sync_to_bytes=409600, sync_from_bytes=1638400)
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

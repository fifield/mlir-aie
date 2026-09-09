"""Independent finite-phase oracle, native binding and fail-stop CPU gates."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from sppelan.phase_alias_host import PhaseAlias, A_ELEMENTS, B_ELEMENTS, STRIPES, ELEMENTS, SHAPE, FRAME_BYTES, frame_input, oracle, validate_output
from sppelan.test_phase_alias import checked_frame, validate_counts, save_failure


class PhaseAliasHostTests(unittest.TestCase):
    def counts(self):
        return dict(run_calls=1, returned_runs=1, completed_runs=1, load_calls=0,
                    sync_to_calls=1, sync_from_calls=1,
                    sync_to_bytes=435200, sync_from_bytes=435200)

    def test_oracle_independent_phase_slices_all_words(self):
        self.assertEqual((A_ELEMENTS, B_ELEMENTS, STRIPES, ELEMENTS, FRAME_BYTES),
                         (12800, 8192, 25, 217600, 435200))
        for frame in range(6):
            bits = frame_input(frame)[1]
            independent = bits.copy()
            independent[:12800] ^= np.uint16(0xA5A5)
            for stripe in range(25):
                start = 12800 + stripe * 8192
                independent[start:start + 8192] ^= np.uint16(0x5A00 | (stripe + 1))
            np.testing.assert_array_equal(oracle(bits), independent)
            self.assertFalse(np.shares_memory(oracle(bits), bits))

    def test_paired_sentinels_identify_every_source_word(self):
        low, high = frame_input(0)[1], frame_input(1)[1]
        self.assertLess(np.unique(low).size, ELEMENTS)
        self.assertLess(np.unique(high).size, ELEMENTS)
        np.testing.assert_array_equal(low.astype(np.uint32) | (high.astype(np.uint32) << 16),
                                      np.arange(ELEMENTS))
        tags = np.concatenate([np.full(12800, 0xA5A5, np.uint16)] +
                              [np.full(8192, 0x5A00 | (s + 1), np.uint16) for s in range(25)])
        recovered = (oracle(low) ^ tags).astype(np.uint32) | ((oracle(high) ^ tags).astype(np.uint32) << 16)
        np.testing.assert_array_equal(recovered, np.arange(ELEMENTS))

    def test_all_phase_and_stripe_boundaries_rejected(self):
        expected = oracle(frame_input(2)[1])
        addresses = [0, 12799, ELEMENTS - 1]
        for stripe in range(25):
            addresses.extend([12800 + stripe * 8192, 12800 + (stripe + 1) * 8192 - 1])
        for address in addresses:
            wrong = expected.copy()
            wrong[address] ^= np.uint16(1)
            with self.assertRaisesRegex(RuntimeError, f'address={address}, phase='):
                validate_output(wrong, expected)

    def test_stale_phase_stripe_reorder_wrong_tags_and_bypass_rejected(self):
        bits = frame_input(2)[1]
        expected = oracle(bits)
        stale = oracle(frame_input(8)[1])
        mutations = [bits, bits ^ np.uint16(0xA5A5), stale]
        phase_stale = expected.copy()
        phase_stale[:12800] = stale[:12800]
        mutations.append(phase_stale)
        stripes = expected.copy()
        stripes[12800:] = np.roll(expected[12800:].reshape(25, 8192), 1, axis=0).reshape(-1)
        mutations.append(stripes)
        reused = expected.copy()
        reused[12800 + 8192:] = np.tile(expected[12800:12800 + 8192], 24)
        mutations.append(reused)
        for wrong in mutations:
            with self.assertRaises(RuntimeError):
                validate_output(wrong, expected)

    def test_patterns_change_and_oracle_preserves_input(self):
        for frame in range(12):
            bits = frame_input(frame)[1]
            before = bits.copy()
            oracle(bits)
            np.testing.assert_array_equal(bits, before)
        self.assertFalse(np.array_equal(frame_input(2)[1], frame_input(8)[1]))
        self.assertFalse(np.array_equal(frame_input(5)[1], frame_input(11)[1]))
        self.assertIn(0x7fc1, frame_input(5)[1])

    def test_invalid_contracts(self):
        for bits in (np.zeros((25, 8192), np.uint16), np.zeros(SHAPE, np.float32)):
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
            for name in ('phase_alias.xclbin', 'phase_alias.bin'):
                Path(directory, name).touch()
            probe = PhaseAlias(directory, backend)
        return probe, backend

    def test_native_persistent_binding_no_phase_sync(self):
        probe, backend = self.make_probe()
        for frame in range(2):
            bits = frame_input(frame)[1]
            before = bits.copy()
            out = probe.run(bits)
            np.testing.assert_array_equal(probe.input.data, bits)
            np.testing.assert_array_equal(bits, before)
            self.assertFalse(np.shares_memory(out, probe.output.data))
        self.assertEqual(backend.iron.zeros.call_count, 2)
        backend.DefaultNPURuntime.load.assert_called_once()
        self.assertEqual(backend.DefaultNPURuntime.run.call_count, 2)
        self.assertEqual(probe.input._sync_to_device.call_count, 2)
        self.assertEqual(probe.output.numpy.call_count, 2)
        backend.DefaultNPURuntime.run.assert_called_with(probe.handle, [probe.input, probe.output])

    def test_runtime_failures_poison_without_retry(self):
        for stage in ('upload', 'run', 'result', 'readback', 'contract'):
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

    def test_count_and_semantic_failure_poison_and_retain_observation(self):
        for kind in ('counts', 'mismatch'):
            probe, backend = self.make_probe()
            bits = frame_input(2)[1]
            expected = oracle(bits)
            probe.output.data[:] = expected
            counts = self.counts()
            if kind == 'counts':
                counts['run_calls'] = 2
            else:
                probe.output.data[12800] ^= np.uint16(1)
            metrics = Mock()
            metrics.return_value.__enter__ = Mock(return_value=Mock(snapshot=Mock(return_value=counts)))
            metrics.return_value.__exit__ = Mock(return_value=False)
            record = {}
            with self.assertRaises(RuntimeError):
                checked_frame(probe, bits, expected, record, metrics_class=metrics)
            self.assertTrue(probe.failed)
            self.assertIn('observed', record)
            with self.assertRaisesRegex(RuntimeError, 'invalid after failure'):
                probe.run(bits)
            self.assertEqual(backend.DefaultNPURuntime.run.call_count, 1)

    def test_missing_artifacts_do_not_load(self):
        backend = Mock()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                PhaseAlias(directory, backend)
        backend.DefaultNPURuntime.load.assert_not_called()

    def test_strict_counts_and_exclusive_evidence(self):
        counts = self.counts()
        validate_counts(counts)
        for key in counts:
            bad = dict(counts)
            bad[key] += 1
            with self.assertRaises(RuntimeError):
                validate_counts(bad)
            del bad[key]
            with self.assertRaises(RuntimeError):
                validate_counts(bad)
        with tempfile.TemporaryDirectory() as directory:
            bits = frame_input(0)[1]
            expected = oracle(bits)
            path = save_failure(directory, 0, 'index', bits, expected, expected, 'failure')
            with np.load(path) as saved:
                for key, value in [('input', bits), ('expected', expected), ('observed', expected)]:
                    np.testing.assert_array_equal(saved[key], value)
                self.assertEqual(str(saved['error']), 'failure')
            with self.assertRaises(FileExistsError):
                save_failure(directory, 0, 'index', bits, expected, None, 'later')


if __name__ == '__main__':
    unittest.main()

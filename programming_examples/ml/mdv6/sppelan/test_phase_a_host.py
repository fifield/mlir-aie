"""CPU-only phase-A packing, pooling and fail-stop tests."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from sppelan.phase_a_host import (PhaseAShard, pack_shard_weights, pool_levels, pool_cases,
                                  exact_bits, values, RAW_WEIGHT_ELEMENTS)
from sppelan.test_phase_a_shard import validate_counts


class PhaseAHostTests(unittest.TestCase):
    def test_all_trained_shard_chunk_mappings(self):
        raw = np.arange(RAW_WEIGHT_ELEMENTS, dtype=np.uint16)
        matrix = raw[:32768].reshape(128, 256)
        for shard in range(16):
            packed = pack_shard_weights(raw, shard).reshape(2, 1056)
            self.assertEqual(packed.strides[0] % 64, 0)
            self.assertFalse(packed[:, 1040:].any())
            for block in range(2):
                unpacked = packed[block, :1024].reshape(16, 8, 8).transpose(2, 0, 1).reshape(8, 128)
                np.testing.assert_array_equal(unpacked, matrix[shard * 8:(shard + 1) * 8, block * 128:(block + 1) * 128])
                np.testing.assert_array_equal(packed[block, 1024:1032], raw[32768 + shard * 8:32768 + (shard + 1) * 8])
                np.testing.assert_array_equal(packed[block, 1032:1040], raw[32896 + shard * 8:32896 + (shard + 1) * 8])
            self.assertFalse(np.shares_memory(packed, raw))

    def test_pack_rebinds_mutated_arrays(self):
        raw = np.zeros(RAW_WEIGHT_ELEMENTS, np.uint16)
        old = pack_shard_weights(raw, 0)
        raw[:] = 42
        new = pack_shard_weights(raw, 0).reshape(2, 1056)
        self.assertFalse(old.any())
        self.assertTrue((new[:, :1040] == 42).all())

    def test_bad_weight_contracts(self):
        for raw, shard in ((np.zeros(1, np.uint16), 0), (np.zeros(RAW_WEIGHT_ELEMENTS, np.float32), 0),
                           (np.zeros(RAW_WEIGHT_ELEMENTS, np.uint16), -1), (np.zeros(RAW_WEIGHT_ELEMENTS, np.uint16), 16)):
            with self.assertRaises(ValueError):
                pack_shard_weights(raw, shard)

    def test_pool_oracle_matches_clipped_window_independent_loop(self):
        for name, bits in pool_cases():
            actual = pool_levels(bits)
            cur = values(bits).reshape(20, 20, 8)
            np.testing.assert_array_equal(actual[0], bits)
            for level in range(1, 4):
                expected = np.empty_like(cur)
                for row in range(20):
                    for col in range(20):
                        window = cur[max(0, row - 2):min(20, row + 3),
                                     max(0, col - 2):min(20, col + 3)].reshape(-1, 8)
                        expected[row, col] = window[window.argmax(axis=0), np.arange(8)]
                np.testing.assert_array_equal(actual[level], exact_bits(expected).reshape(400, 8), err_msg=name)
                cur = expected

    def test_pool_oracle_matches_torch_including_signed_zero(self):
        import torch
        for name, bits in pool_cases():
            expected = pool_levels(bits)
            cur = torch.from_numpy(bits.copy()).view(torch.bfloat16).reshape(20, 20, 8).permute(2, 0, 1)[None]
            for level in range(1, 4):
                cur = torch.nn.functional.max_pool2d(cur, 5, stride=1, padding=2)
                actual = cur[0].permute(1, 2, 0).contiguous().view(torch.uint16).numpy().reshape(400, 8)
                np.testing.assert_array_equal(actual, expected[level], err_msg=name)

    def test_negative_padding_and_corner_receptive_field(self):
        negative = exact_bits(np.full((400, 8), -3, np.float32))
        self.assertTrue((values(pool_levels(negative)) == -3).all())
        corner = np.full((20, 20, 8), -4, np.float32)
        corner[0, 0, 3] = 1
        levels = values(pool_levels(exact_bits(corner).reshape(400, 8))).reshape(4, 20, 20, 8)
        for level in range(4):
            self.assertEqual(np.count_nonzero(levels[level, :, :, 3] == 1), (2 * level + 1) ** 2)
            self.assertTrue((levels[level, :, :, :3] == -4).all())
            self.assertTrue((levels[level, :, :, 4:] == -4).all())

    def test_pool_rejects_nonfinite(self):
        bits = np.zeros((400, 8), np.uint16)
        bits[0, 0] = 0x7f80
        with self.assertRaises(ValueError):
            pool_levels(bits)

    def make_probe(self, pool=False):
        backend = Mock()
        def buffer(n, dtype):
            result = Mock(data=np.zeros(n, dtype))
            result.numpy.return_value = result.data
            return result
        backend.iron.zeros.side_effect = buffer
        backend.DefaultNPURuntime.run.return_value.is_success.return_value = True
        with tempfile.TemporaryDirectory() as directory:
            for name in ('phase_a_pool' if pool else 'phase_a_shard',):
                for ext in ('xclbin', 'bin'):
                    Path(directory, f'{name}.{ext}').touch()
            probe = PhaseAShard(directory, backend, pool)
        return probe, backend

    def test_persistent_modes_and_arenas(self):
        for pool in (False, True):
            probe, backend = self.make_probe(pool)
            bits = np.zeros((400, 8 if pool else 256), np.uint16)
            weights = None if pool else np.zeros(2112, np.uint16)
            for _ in range(2):
                out = probe.run(bits, weights)
                self.assertEqual(out.shape, (4, 400, 8))
                self.assertFalse(np.shares_memory(out, probe.output.data))
            backend.DefaultNPURuntime.load.assert_called_once()
            self.assertEqual(backend.iron.zeros.call_count, 3 if pool else 4)
            self.assertEqual(probe.input._sync_to_device.call_count, 2)
            self.assertEqual(probe.output.numpy.call_count, 2)
            self.assertEqual(probe.metadata.numpy.call_count, 2)

    def test_runtime_metadata_and_nonfinite_failures_poison(self):
        for stage in ('upload', 'run', 'result', 'readback', 'metadata', 'nonfinite'):
            probe, backend = self.make_probe()
            if stage == 'upload':
                probe.input._sync_to_device.side_effect = RuntimeError('upload failed')
            elif stage == 'run':
                backend.DefaultNPURuntime.run.side_effect = RuntimeError('run failed')
            elif stage == 'result':
                backend.DefaultNPURuntime.run.return_value.is_success.return_value = False
            elif stage == 'readback':
                probe.output.numpy.side_effect = RuntimeError('readback failed')
            elif stage == 'metadata':
                probe.metadata.data[0] = 8
            else:
                probe.output.data[0] = 0x7f80
            with self.assertRaises(RuntimeError):
                probe.run(np.zeros((400, 256), np.uint16), np.zeros(2112, np.uint16))
            with self.assertRaisesRegex(RuntimeError, 'invalid after failure'):
                probe.run(np.zeros((400, 256), np.uint16), np.zeros(2112, np.uint16))
            self.assertEqual(backend.DefaultNPURuntime.run.call_count, 0 if stage == 'upload' else 1)
            backend.DefaultNPURuntime.load.assert_called_once()

    def test_missing_artifacts_never_load(self):
        backend = Mock()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                PhaseAShard(directory, backend)
        backend.DefaultNPURuntime.load.assert_not_called()

    def test_strict_counts_reject_extra_run_and_sync(self):
        counts = dict(run_calls=1, returned_runs=1, completed_runs=1, load_calls=0,
                      sync_to_calls=2, sync_from_calls=2, sync_to_bytes=209024, sync_from_bytes=25664)
        validate_counts(counts)
        for key in counts:
            broken = dict(counts)
            broken[key] += 1
            with self.assertRaises(RuntimeError):
                validate_counts(broken)


if __name__ == '__main__':
    unittest.main()

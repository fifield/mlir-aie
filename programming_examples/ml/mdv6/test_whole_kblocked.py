"""CPU-only ABI, rebinding and fail-stop checks for K-blocked batching."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from whole_kblocked import (WholeKBlocked, pack_input, pack_weights, unpack_output,
                            INPUT_ELEMENTS, OUTPUT_ELEMENTS, RAW_WEIGHT_ELEMENTS,
                            WEIGHT_ELEMENTS)


class LayoutTests(unittest.TestCase):
    def test_zero_row_semantic_diagnostic(self):
        from gemm_conv1x1.test_whole_kblocked import zero_row_diagnostic
        x = np.zeros((80, 80, 256), np.float32)
        out = np.zeros((80, 80, 128), np.float32)
        weights = np.zeros(RAW_WEIGHT_ELEMENTS, np.uint16)
        x[0, 0, 0] = 1
        out[0, 0, 0] = 99  # Nonzero input row is not part of the invariant.
        self.assertEqual(zero_row_diagnostic(x, out, weights)['nonzero_output_elements'], 0)
        out[0, 1, :2] = [2, -3]
        d = zero_row_diagnostic(x, out, weights)
        self.assertEqual(d, dict(applicable=True, zero_input_rows=6399,
                                 nonzero_output_rows=1, nonzero_output_elements=2, max_abs_output=3.0))
        weights[-128:] = 0x8000  # Negative zero BN bias still qualifies.
        self.assertTrue(zero_row_diagnostic(x, out, weights)['applicable'])
        weights[-1] = 0x3f80
        self.assertFalse(zero_row_diagnostic(x, out, weights)['applicable'])

    def test_input_matches_baseline_slots(self):
        bits = np.arange(80 * 80 * 256, dtype=np.uint16).reshape(80, 80, 256)
        arena = pack_input(bits).reshape(3, 32, 68, 256)
        flat = bits.reshape(-1, 256)
        for batch in range(3):
            for slot in range(32):
                start = (batch * 32 + slot) * 68
                if start >= 6400:
                    np.testing.assert_array_equal(arena[batch, slot], arena[batch, 0])
                else:
                    rows = min(68, 6400 - start)
                    np.testing.assert_array_equal(arena[batch, slot, :rows], flat[start:start + rows])
                    self.assertFalse(arena[batch, slot, rows:].any())
        np.testing.assert_array_equal(arena[2, 30, :8], flat[-8:])
        self.assertFalse(arena[2, 30, 8:].any())
        self.assertFalse(np.shares_memory(arena, bits))

    def test_each_weight_chunk_and_repeated_bn(self):
        bits = np.arange(RAW_WEIGHT_ELEMENTS, dtype=np.uint16)
        packed = pack_weights(bits).reshape(16, 2304)
        matrix = bits[:32768].reshape(128, 256)
        for block in range(16):
            recovered = packed[block, :2048].reshape(2, 16, 8, 8).transpose(1, 3, 0, 2).reshape(128, 16)
            np.testing.assert_array_equal(recovered, matrix[:, block * 16:(block + 1) * 16])
            np.testing.assert_array_equal(packed[block, 2048:], bits[32768:])
        self.assertFalse(np.shares_memory(packed, bits))

    def test_weight_mutation_repacked(self):
        bits = np.zeros(RAW_WEIGHT_ELEMENTS, np.uint16)
        first = pack_weights(bits)
        bits[:] = 42
        second = pack_weights(bits)
        self.assertFalse(first.any())
        self.assertTrue((second == 42).all())

    def test_output_crop_owns_data(self):
        bits = np.arange(OUTPUT_ELEMENTS, dtype=np.uint16)
        output = unpack_output(bits)
        self.assertEqual(output.shape, (80, 80, 128))
        np.testing.assert_array_equal(output.reshape(-1), bits[:6400 * 128])
        self.assertFalse(np.shares_memory(output, bits))

    def test_bad_contracts(self):
        for bits in (np.zeros((80, 80, 128), np.uint16), np.zeros((80, 80, 256), np.float32)):
            with self.assertRaises(ValueError):
                pack_input(bits)
        for bits in (np.zeros(RAW_WEIGHT_ELEMENTS - 1, np.uint16), np.zeros(RAW_WEIGHT_ELEMENTS, np.float32)):
            with self.assertRaises(ValueError):
                pack_weights(bits)
        for bits in (np.zeros(OUTPUT_ELEMENTS - 1, np.uint16), np.zeros(OUTPUT_ELEMENTS, np.float32)):
            with self.assertRaises(ValueError):
                unpack_output(bits)

    def test_missing_artifacts_do_not_load(self):
        backend = Mock()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                WholeKBlocked(directory, backend)
        backend.DefaultNPURuntime.load.assert_not_called()

    def make_fake(self):
        x = Mock(dtype='bf16', device=SimpleNamespace(type='cpu'))
        x.contiguous.return_value.view.return_value.numpy.return_value = np.zeros((80, 80, 256), np.uint16)
        whole = WholeKBlocked.__new__(WholeKBlocked)
        whole.backend = Mock(torch=SimpleNamespace(bfloat16='bf16', uint16='u16', from_numpy=Mock()))
        whole.failed = False
        whole.handle = object()
        whole.input, whole.weights, whole.output = Mock(), Mock(), Mock()
        whole.output.numpy.return_value = np.zeros(OUTPUT_ELEMENTS, np.uint16)
        whole.backend.DefaultNPURuntime.run.return_value.is_success.return_value = True
        return whole, x

    def test_constructor_arenas(self):
        backend = Mock()
        backend.iron.zeros.side_effect = lambda n, dtype: np.zeros(n, dtype)
        with tempfile.TemporaryDirectory() as directory:
            for name in ('whole_kblocked.xclbin', 'whole_kblocked.bin'):
                Path(directory, name).touch()
            whole = WholeKBlocked(directory, backend)
        self.assertEqual([whole.input.size, whole.weights.size, whole.output.size],
                         [INPUT_ELEMENTS, WEIGHT_ELEMENTS, OUTPUT_ELEMENTS])
        backend.DefaultNPURuntime.load.assert_called_once()
        self.assertEqual(backend.iron.zeros.call_count, 3)

    def test_success_reuses_arenas_and_uploads_changed_weights(self):
        whole, x = self.make_fake()
        weights = np.zeros(RAW_WEIGHT_ELEMENTS, np.uint16)
        whole.run(x, weights)
        weights[:] = 123
        whole.run(x, weights)
        calls = whole.backend._fill_and_sync.call_args_list
        self.assertEqual(len(calls), 4)
        self.assertFalse(calls[1].args[1].any())
        self.assertTrue((calls[3].args[1] == 123).all())
        self.assertEqual(whole.output.numpy.call_count, 2)
        self.assertEqual(whole.backend.DefaultNPURuntime.run.call_count, 2)
        whole.backend.DefaultNPURuntime.run.assert_called_with(whole.handle, [whole.input, whole.weights, whole.output])
        self.assertFalse(whole.failed)

    def test_upload_launch_unsuccessful_result_readback_invalidate_without_retry(self):
        for stage in ('upload', 'launch', 'result', 'readback'):
            with self.subTest(stage=stage):
                whole, x = self.make_fake()
                if stage == 'upload':
                    whole.backend._fill_and_sync.side_effect = RuntimeError('upload failed')
                elif stage == 'launch':
                    whole.backend.DefaultNPURuntime.run.side_effect = RuntimeError('launch failed')
                elif stage == 'result':
                    whole.backend.DefaultNPURuntime.run.return_value.is_success.return_value = False
                else:
                    whole.output.numpy.side_effect = RuntimeError('readback failed')
                with self.assertRaises(RuntimeError):
                    whole.run(x, np.zeros(RAW_WEIGHT_ELEMENTS, np.uint16))
                with self.assertRaisesRegex(RuntimeError, 'invalid after failed run'):
                    whole.run(x, np.zeros(RAW_WEIGHT_ELEMENTS, np.uint16))
                self.assertTrue(whole.failed)
                self.assertEqual(whole.backend.DefaultNPURuntime.run.call_count, 0 if stage == 'upload' else 1)
                whole.backend.DefaultNPURuntime.load.assert_not_called()

    def test_bad_tensor_contract_before_upload_does_not_poison(self):
        for dtype, device in (('fp32', 'cpu'), ('bf16', 'npu')):
            whole, x = self.make_fake()
            x.dtype, x.device.type = dtype, device
            with self.assertRaises(ValueError):
                whole.run(x, np.zeros(RAW_WEIGHT_ELEMENTS, np.uint16))
            whole.backend._fill_and_sync.assert_not_called()
            self.assertFalse(whole.failed)


if __name__ == '__main__':
    unittest.main()

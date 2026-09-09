"""Dependency-light CPU checks of whole-GEMM ABI and failure behavior."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from whole_gemm import (WholeGemm, pack_input, pack_weights, unpack_output,
                        INPUT_ELEMENTS, OUTPUT_ELEMENTS, WEIGHT_ELEMENTS)


class LayoutTests(unittest.TestCase):
    def test_input_matches_baseline_slots(self):
        bits = np.arange(160 * 160 * 128, dtype=np.uint16).reshape(160, 160, 128)
        arena = pack_input(bits).reshape(4, 64, 104, 128)
        flat = bits.reshape(-1, 128)
        for batch in range(4):
            for slot in range(64):
                start = (batch * 64 + slot) * 104
                if start >= 25600:
                    np.testing.assert_array_equal(arena[batch, slot], arena[batch, 0])
                else:
                    rows = min(104, 25600 - start)
                    np.testing.assert_array_equal(arena[batch, slot, :rows], flat[start:start + rows])
                    self.assertFalse(arena[batch, slot, rows:].any())

    def test_weight_layout_and_bn(self):
        bits = np.arange(WEIGHT_ELEMENTS, dtype=np.uint16)
        packed = pack_weights(bits)
        recovered = packed[:8192].reshape(16, 8, 8, 8).transpose(1, 3, 0, 2).reshape(64, 128)
        np.testing.assert_array_equal(recovered, bits[:8192].reshape(64, 128))
        np.testing.assert_array_equal(packed[8192:], bits[8192:])

    def test_output_crop_owns_data(self):
        bits = np.arange(OUTPUT_ELEMENTS, dtype=np.uint16)
        output = unpack_output(bits)
        self.assertEqual(output.shape, (160, 160, 64))
        np.testing.assert_array_equal(output.reshape(-1), bits[:25600 * 64])
        self.assertFalse(np.shares_memory(output, bits))

    def test_input_does_not_alias(self):
        bits = np.zeros((160, 160, 128), np.uint16)
        self.assertFalse(np.shares_memory(pack_input(bits), bits))

    def test_bad_contracts(self):
        for bits in (np.zeros((80, 80, 128), np.uint16), np.zeros((160, 160, 128), np.float32)):
            with self.assertRaises(ValueError):
                pack_input(bits)
        for bits in (np.zeros(WEIGHT_ELEMENTS - 1, np.uint16), np.zeros(WEIGHT_ELEMENTS, np.float32)):
            with self.assertRaises(ValueError):
                pack_weights(bits)
        for bits in (np.zeros(OUTPUT_ELEMENTS - 1, np.uint16), np.zeros(OUTPUT_ELEMENTS, np.float32)):
            with self.assertRaises(ValueError):
                unpack_output(bits)

    def test_missing_artifacts_do_not_load(self):
        backend = Mock()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                WholeGemm(directory, backend)
        backend.DefaultNPURuntime.load.assert_not_called()

    def test_persistent_arenas_and_failure_no_retry(self):
        # Minimal torch/tensor stand-in keeps these tests independent of torch/XRT.
        data = np.zeros((160, 160, 128), np.uint16)
        x = Mock(dtype='bf16', device=SimpleNamespace(type='cpu'))
        x.contiguous.return_value.view.return_value.numpy.return_value = data
        backend = Mock(torch=SimpleNamespace(bfloat16='bf16', uint16='u16'))
        backend.iron.zeros.side_effect = lambda n, dtype: np.zeros(n, dtype)
        with tempfile.TemporaryDirectory() as directory:
            for name in ('whole_gemm.xclbin', 'whole_gemm.bin'):
                Path(directory, name).touch()
            whole = WholeGemm(directory, backend)
        self.assertEqual([whole.input.size, whole.weights.size, whole.output.size],
                         [INPUT_ELEMENTS, WEIGHT_ELEMENTS, OUTPUT_ELEMENTS])
        backend.DefaultNPURuntime.run.side_effect = RuntimeError('device failed')
        with self.assertRaisesRegex(RuntimeError, 'device failed'):
            whole.run(x, np.zeros(WEIGHT_ELEMENTS, np.uint16))
        with self.assertRaisesRegex(RuntimeError, 'invalid after failed run'):
            whole.run(x, np.zeros(WEIGHT_ELEMENTS, np.uint16))
        backend.DefaultNPURuntime.run.assert_called_once()
        backend.DefaultNPURuntime.load.assert_called_once()
        self.assertEqual(backend.iron.zeros.call_count, 3)

    def test_success_reuses_arenas_and_uploads_once_per_call(self):
        data = np.zeros((160, 160, 128), np.uint16)
        x = Mock(dtype='bf16', device=SimpleNamespace(type='cpu'))
        x.contiguous.return_value.view.return_value.numpy.return_value = data
        torch = SimpleNamespace(bfloat16='bf16', uint16='u16', from_numpy=Mock())
        whole = WholeGemm.__new__(WholeGemm)
        whole.backend = Mock(torch=torch)
        whole.failed = False
        whole.handle = object()
        whole.input, whole.weights, whole.output = Mock(), Mock(), Mock()
        whole.output.numpy.return_value = np.zeros(OUTPUT_ELEMENTS, np.uint16)
        whole.backend.DefaultNPURuntime.run.return_value.is_success.return_value = True
        arenas = [whole.input, whole.weights, whole.output]
        for _ in range(2):
            whole.run(x, np.zeros(WEIGHT_ELEMENTS, np.uint16))
        self.assertEqual(whole.backend._fill_and_sync.call_count, 4)
        self.assertEqual(whole.output.numpy.call_count, 2)
        self.assertEqual(whole.backend.DefaultNPURuntime.run.call_count, 2)
        whole.backend.DefaultNPURuntime.run.assert_called_with(whole.handle, arenas)
        self.assertFalse(whole.failed)

    def test_upload_unsuccessful_result_and_readback_invalidate(self):
        for stage in ('upload', 'result', 'readback'):
            with self.subTest(stage=stage):
                x = Mock(dtype='bf16', device=SimpleNamespace(type='cpu'))
                x.contiguous.return_value.view.return_value.numpy.return_value = np.zeros((160, 160, 128), np.uint16)
                whole = WholeGemm.__new__(WholeGemm)
                whole.backend = Mock(torch=SimpleNamespace(bfloat16='bf16', uint16='u16', from_numpy=Mock()))
                whole.failed = False
                whole.handle = object()
                whole.input, whole.weights, whole.output = Mock(), Mock(), Mock()
                whole.backend.DefaultNPURuntime.run.return_value.is_success.return_value = stage != 'result'
                if stage == 'upload':
                    whole.backend._fill_and_sync.side_effect = RuntimeError('upload failed')
                else:
                    whole.output.numpy.side_effect = RuntimeError('readback failed')
                for _ in range(2):
                    with self.assertRaises(RuntimeError):
                        whole.run(x, np.zeros(WEIGHT_ELEMENTS, np.uint16))
                self.assertTrue(whole.failed)
                self.assertEqual(whole.backend.DefaultNPURuntime.run.call_count, 0 if stage == 'upload' else 1)


if __name__ == '__main__':
    unittest.main()

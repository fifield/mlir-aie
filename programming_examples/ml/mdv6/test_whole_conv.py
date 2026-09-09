"""CPU-only physical layout and boundary tests for whole-operator arenas."""
import unittest
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from whole_conv import pack_input, pack_weights, unpack_output, WEIGHT_SLOT, OUTPUT_ELEMENTS


class LayoutTests(unittest.TestCase):
    def route_function(self, directory):
        tree = ast.parse(Path(__file__).with_name('run_tiled_mc.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_tiled_fused_conv_mc')
        namespace = dict(_whole_conv_dir=directory, _whole_conv=None, torch=None,
                         iron=None, NPUKernel=None, DefaultNPURuntime=None,
                         _fill_and_sync=None, _get_mc_variant=Mock(side_effect=RuntimeError('baseline route')))
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<route>', 'exec'), namespace)
        return namespace

    def test_opt_in_exact_shape_and_executor_reuse(self):
        ns = self.route_function('/experimental')
        x = SimpleNamespace(shape=(20, 20, 64))
        with patch('whole_conv.WholeConv') as factory:
            factory.return_value.run.return_value = 'result'
            for _ in range(2):
                self.assertEqual(ns['run_tiled_fused_conv_mc']('mc_re8_rn3', '', x,
                    None, 20, 20, 64, 8, 8, 16), 'result')
            factory.assert_called_once()
        ns['_get_mc_variant'].assert_not_called()

    def test_wrong_shape_rejected_before_loading(self):
        ns = self.route_function('/experimental')
        with self.assertRaises(ValueError):
            ns['run_tiled_fused_conv_mc']('mc_re8_rn3', '', SimpleNamespace(shape=(20, 20, 32)),
                                         None, 20, 20, 64, 8, 8, 16)
        ns['_get_mc_variant'].assert_not_called()

    def test_unset_flag_keeps_baseline(self):
        ns = self.route_function(None)
        with self.assertRaisesRegex(RuntimeError, 'baseline route'):
            ns['run_tiled_fused_conv_mc']('mc_re8_rn3', '', None, None, 20, 20, 64, 8, 8, 16)
        ns['_get_mc_variant'].assert_called_once()

    def test_patch_halos_and_duplicate_unused_cores(self):
        x = np.arange(20 * 20 * 64, dtype=np.uint16).reshape(20, 20, 64)
        patches = pack_input(x).reshape(32, 10, 10, 64)
        for i in range(9):
            r, c = divmod(i, 3)
            for y in range(10):
                for z in range(10):
                    iy, ix = r * 8 + y - 1, c * 8 + z - 1
                    expected = x[iy, ix] if 0 <= iy < 20 and 0 <= ix < 20 else np.zeros(64, np.uint16)
                    np.testing.assert_array_equal(patches[i, y, z], expected)
        np.testing.assert_array_equal(patches[9:], np.repeat(patches[:1], 23, axis=0))

    def test_weight_order_and_bn_blocks(self):
        raw = np.arange(64 * 64 * 9 + 128, dtype=np.uint16)
        packed = pack_weights(raw).reshape(4, WEIGHT_SLOT)
        original = raw[:64 * 64 * 9].reshape(64, 64, 9)
        for block in range(4):
            recovered = packed[block, :-32].reshape(2, 8, 9, 8, 8).transpose(0, 4, 1, 3, 2).reshape(16, 64, 9)
            np.testing.assert_array_equal(recovered, original[block * 16:(block + 1) * 16])
            np.testing.assert_array_equal(packed[block, -32:-16], raw[64 * 64 * 9 + block * 16:64 * 64 * 9 + (block + 1) * 16])
            np.testing.assert_array_equal(packed[block, -16:], raw[-64:][block * 16:(block + 1) * 16])

    def test_output_oc_order_and_partial_crop(self):
        arena = np.empty((4, 32, 8, 8, 16), np.uint16)
        for b in range(4):
            for i in range(32):
                arena[b, i] = b * 100 + i
        output = unpack_output(arena.reshape(-1))
        for b in range(4):
            for y in range(20):
                for x in range(20):
                    np.testing.assert_array_equal(output[y, x, b * 16:(b + 1) * 16], b * 100 + (y // 8) * 3 + x // 8)

    def test_reject_bad_shapes_and_dtypes(self):
        with self.assertRaises(ValueError):
            pack_input(np.zeros((20, 20, 64), np.float32))
        with self.assertRaises(ValueError):
            pack_weights(np.zeros(1, np.uint16))
        with self.assertRaises(ValueError):
            unpack_output(np.zeros(OUTPUT_ELEMENTS - 1, np.uint16))


if __name__ == '__main__':
    unittest.main()

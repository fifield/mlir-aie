"""Device-free AST tests of the shape-specific experimental routing guard."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class RouteTests(unittest.TestCase):
    def route(self, enabled=True):
        tree = ast.parse(Path(__file__).with_name('run_tiled_mc.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_gemm_conv1x1_mc')
        ns = dict(_whole_gemm_dir='/experimental' if enabled else None, _whole_gemm=None,
                  torch=None, iron=None, NPUKernel=None, DefaultNPURuntime=None, _fill_and_sync=None,
                  _gemm_choose_k_block=Mock(side_effect=RuntimeError('baseline path')))
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<route>', 'exec'), ns)
        return ns

    def call(self, ns, h=160, w=160, ic=128, oc=64, name='gemm_elan_c4', ob=None):
        return ns['run_gemm_conv1x1_mc'](name, '', SimpleNamespace(shape=(h, w, ic)),
                                        None, h, w, oc, ob)

    def test_exact_shape_routes_and_reuses_executor(self):
        ns = self.route()
        with patch('whole_gemm.WholeGemm') as factory:
            factory.return_value.run.return_value = 'result'
            self.assertEqual(self.call(ns), 'result')
            self.assertEqual(self.call(ns), 'result')
            factory.assert_called_once()
        ns['_gemm_choose_k_block'].assert_not_called()

    def test_eighty_pixel_alias_keeps_original_route(self):
        ns = self.route()
        with self.assertRaisesRegex(RuntimeError, 'baseline path'):
            self.call(ns, h=80, w=80, ic=64)
        self.assertIsNone(ns['_whole_gemm'])

    def test_unset_and_unrelated_names_keep_original_route(self):
        for enabled, name in ((False, 'gemm_elan_c4'), (True, 'gemm_other')):
            ns = self.route(enabled)
            with self.assertRaisesRegex(RuntimeError, 'baseline path'):
                self.call(ns, name=name)

    def test_rejects_incompatible_contract_before_loading(self):
        for kwargs in (dict(ic=64), dict(oc=128), dict(ob=32)):
            ns = self.route()
            with self.assertRaises(ValueError):
                self.call(ns, **kwargs)
            ns['_gemm_choose_k_block'].assert_not_called()


if __name__ == '__main__':
    unittest.main()

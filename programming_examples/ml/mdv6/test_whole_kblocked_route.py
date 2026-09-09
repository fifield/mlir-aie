"""Device-free exact-shape and failure routing checks for KB batching."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class RouteTests(unittest.TestCase):
    def route(self, enabled=True):
        tree = ast.parse(Path(__file__).with_name('run_tiled_mc.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_gemm_conv1x1_mc')
        ns = dict(_whole_kblocked_dir='/experimental' if enabled else None,
                  _whole_kblocked=None, _whole_gemm_dir=None, _whole_gemm=None,
                  torch=None, iron=None, NPUKernel=None, DefaultNPURuntime=None,
                  _fill_and_sync=None,
                  _gemm_choose_k_block=Mock(side_effect=RuntimeError('baseline path')))
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<route>', 'exec'), ns)
        return ns

    def call(self, ns, shape=(80, 80, 256), oc=128, name='gemm_re4_c4', ob=None):
        return ns['run_gemm_conv1x1_mc'](name, '', SimpleNamespace(shape=shape),
                                        None, 80, 80, oc, ob)

    def test_routes_and_reuses_executor(self):
        ns = self.route()
        with patch('whole_kblocked.WholeKBlocked') as factory:
            factory.return_value.run.return_value = 'result'
            for _ in range(2):
                self.assertEqual(self.call(ns), 'result')
            factory.assert_called_once()
        ns['_gemm_choose_k_block'].assert_not_called()

    def test_failures_do_not_fallback(self):
        ns = self.route()
        with patch('whole_kblocked.WholeKBlocked') as factory:
            factory.return_value.run.side_effect = RuntimeError('device failure')
            with self.assertRaisesRegex(RuntimeError, 'device failure'):
                self.call(ns)
        ns['_gemm_choose_k_block'].assert_not_called()

    def test_unset_flag_and_other_names_keep_original_route(self):
        for enabled, name in ((False, 'gemm_re4_c4'), (True, 'gemm_other')):
            ns = self.route(enabled)
            with self.assertRaisesRegex(RuntimeError, 'baseline path'):
                self.call(ns, name=name)

    def test_incompatible_contract_rejected_before_loading(self):
        for kwargs in (dict(shape=(40, 40, 256)), dict(shape=(80, 80, 128)), dict(oc=64), dict(ob=16)):
            ns = self.route()
            with self.assertRaises(ValueError):
                self.call(ns, **kwargs)
            ns['_gemm_choose_k_block'].assert_not_called()


if __name__ == '__main__':
    unittest.main()

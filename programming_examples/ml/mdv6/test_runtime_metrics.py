"""No XRT/device imports: validate counters and patch restoration."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from runtime_metrics import RuntimeMetrics


class Tensor:
    def buffer_object(self):
        return SimpleNamespace(size=lambda: 64)

    def _sync_to_device(self):
        pass

    def _sync_from_device(self):
        pass


class Runtime:
    def run(self, fail=False):
        if fail:
            raise ValueError('device failure')
        return SimpleNamespace(is_success=lambda: True)

    def load(self, kernel):
        return kernel


class MetricsTests(unittest.TestCase):
    def test_cache_misses_are_not_load_calls(self):
        class CachedRuntime(Runtime):
            def __init__(self):
                self._context_cache = {}

            def load(self, kernel):
                path = Path(kernel.xclbin_path).resolve()
                self._context_cache[(str(path), path.stat().st_mtime)] = object()

            def _evict(self):
                self._context_cache.clear()

        runtime = CachedRuntime()
        with tempfile.NamedTemporaryFile() as artifact:
            kernel = SimpleNamespace(xclbin_path=artifact.name)
            with RuntimeMetrics(runtime, Tensor) as metrics:
                runtime.load(kernel)
                runtime.load(kernel)
                runtime._evict()
                runtime.load(kernel)
                row = metrics.snapshot()
                self.assertEqual(row['load_calls'], 3)
                self.assertEqual(row['context_cache_misses'], 2)
                self.assertEqual(row['eviction_calls'], 1)

    def test_counts_and_restores_inherited_methods(self):
        runtime = Runtime()
        original = Tensor._sync_to_device
        with RuntimeMetrics(runtime, Tensor) as metrics:
            runtime.run()
            runtime.load(None)
            Tensor()._sync_to_device()
            Tensor()._sync_from_device()
            row = metrics.snapshot()
            self.assertEqual(row['completed_runs'], 1)
            self.assertEqual(row['load_calls'], 1)
            self.assertEqual(row['sync_to_bytes'], 64)
            self.assertEqual(row['sync_from_bytes'], 64)
            self.assertIsNone(row['context_cache_misses'])
            self.assertIsNone(row['device_dma_bytes'])
        self.assertNotIn('run', vars(runtime))
        self.assertIs(Tensor._sync_to_device, original)

    def test_failure_is_not_completed(self):
        runtime = Runtime()
        with self.assertRaises(ValueError):
            with RuntimeMetrics(runtime, Tensor) as metrics:
                runtime.run(fail=True)
        self.assertEqual(metrics.snapshot()['run_calls'], 1)
        self.assertEqual(metrics.snapshot()['completed_runs'], 0)
        self.assertNotIn('run', vars(runtime))

    def test_nesting_rejected_and_original_observer_survives(self):
        runtime = Runtime()
        with RuntimeMetrics(runtime, Tensor) as metrics:
            with self.assertRaises(RuntimeError):
                with RuntimeMetrics(runtime, Tensor):
                    pass
            runtime.run()
            self.assertEqual(metrics.snapshot()['completed_runs'], 1)

    def test_partial_install_restored(self):
        runtime = Runtime()
        with self.assertRaises(AttributeError):
            with RuntimeMetrics(runtime, object):
                pass
        self.assertNotIn('run', vars(runtime))
        with RuntimeMetrics(runtime, Tensor):
            runtime.run()


if __name__ == '__main__':
    unittest.main()

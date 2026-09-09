"""CPU-only validation-loop tests: python -m unittest test_validate_stream."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import validate_stream as stream


class ValidateStreamTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.args = SimpleNamespace(frames=3, seed=42, class_tolerance=0.5,
                                    vector_tolerance=0.1,
                                    report=Path(self.temp.name) / 'frames.jsonl')
        self.original_run = Mock()
        self.runtime = SimpleNamespace(run=self.original_run, _context_cache={'context': object()})
        self.model = SimpleNamespace(mcr=SimpleNamespace(cached_kernel_inventory=lambda: [
            {'xclbin': '/build/a.xclbin', 'instructions': '/build/a_1.bin'},
            {'xclbin': '/build/a.xclbin', 'instructions': '/build/a_2.bin'},
            {'xclbin': '/build/b.xclbin', 'instructions': '/build/b.bin'},
        ]))
        self.seeds = []
        self.addCleanup(patch.stopall)
        patch.object(stream, 'resident_bytes', return_value=1234).start()
        patch.object(stream.gc, 'collect').start()

    def model_call(self, *, seed, class_tolerance, vector_tolerance, metrics):
        self.seeds.append(seed)
        self.assertEqual((class_tolerance, vector_tolerance), (0.5, 0.1))
        self.runtime.run('first')
        self.runtime.run('second')
        metrics.update(finite=True, max_class_diff=0.2, max_vector_diff=0.03, ok=True)
        return True

    def run_frames(self, callback=None):
        self.model.main = callback or self.model_call
        with contextlib.redirect_stdout(io.StringIO()):
            return stream.run_frames(self.args, self.model, self.runtime)

    def rows(self):
        return [json.loads(line) for line in self.args.report.read_text().splitlines()]

    def test_success_increments_seeds_and_counts_each_frame(self):
        self.assertEqual(self.run_frames(), 0)
        self.assertEqual(self.seeds, [42, 43, 44])
        rows = self.rows()
        self.assertEqual([r['frame'] for r in rows], [0, 1, 2])
        for row in rows:
            self.assertTrue(row['ok'])
            self.assertEqual(row['launches'], 2)
            self.assertEqual(row['cached_handles'], 3)
            self.assertEqual(row['distinct_artifacts'], 2)
            self.assertEqual(row['runtime_cached_contexts'], 1)
            self.assertEqual(row['rss_bytes'], 1234)
        self.assertIs(self.runtime.run, self.original_run)

    def test_false_stops_after_first_failed_frame(self):
        def callback(**kwargs):
            self.model_call(**kwargs)
            passed = kwargs['seed'] == 42
            kwargs['metrics']['ok'] = passed
            return passed

        self.assertEqual(self.run_frames(callback), 1)
        self.assertEqual(self.seeds, [42, 43])
        self.assertEqual([r['ok'] for r in self.rows()], [True, False])
        self.assertIs(self.runtime.run, self.original_run)

    def test_exception_writes_failed_row_and_restores_runtime(self):
        def callback(**kwargs):
            self.seeds.append(kwargs['seed'])
            self.runtime.run('attempt')
            raise RuntimeError('simulated timeout')

        with self.assertRaisesRegex(RuntimeError, 'simulated timeout'):
            self.run_frames(callback)
        row, = self.rows()
        self.assertFalse(row['ok'])
        self.assertEqual(row['error'], 'RuntimeError: simulated timeout')
        self.assertEqual(row['launches'], 1)
        self.assertEqual(self.seeds, [42])
        self.assertIs(self.runtime.run, self.original_run)

    def test_completed_rows_are_flushed_before_next_frame(self):
        def callback(**kwargs):
            self.assertEqual(len(self.rows()), kwargs['seed'] - 42)
            return self.model_call(**kwargs)

        self.assertEqual(self.run_frames(callback), 0)

    def test_existing_report_is_preserved(self):
        self.assertEqual(self.run_frames(), 0)
        before = self.args.report.read_bytes()
        self.seeds.clear()
        with self.assertRaises(FileExistsError):
            self.run_frames()
        self.assertEqual(self.args.report.read_bytes(), before)
        self.assertEqual(self.seeds, [])
        self.assertIs(self.runtime.run, self.original_run)

    def test_early_success_without_comparison_metrics_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'metrics are incomplete'):
            self.run_frames(lambda **kwargs: True)
        row, = self.rows()
        self.assertFalse(row['ok'])
        self.assertIn('metrics are incomplete', row['error'])
        self.assertIs(self.runtime.run, self.original_run)

    def test_metrics_failure_cannot_be_overridden_by_return_value(self):
        def callback(**kwargs):
            self.model_call(**kwargs)
            kwargs['metrics']['ok'] = False
            return True

        self.assertEqual(self.run_frames(callback), 1)
        self.assertFalse(self.rows()[0]['ok'])

    def test_context_count_unavailable(self):
        result = stream.handle_inventory(self.model.mcr, SimpleNamespace())
        self.assertIsNone(result['runtime_cached_contexts'])

    def test_invalid_cli_options_fail_before_runtime_import(self):
        base = ['--frames', '1', '--report', str(self.args.report)]
        with patch.dict('os.environ', {}, clear=True):
            for option in ('--class-tolerance', '--vector-tolerance'):
                for value in ('nan', 'inf', '-inf', '0', '-0.1'):
                    with self.subTest(option=option, value=value):
                        with contextlib.redirect_stderr(io.StringIO()):
                            with self.assertRaises(SystemExit) as caught:
                                stream.main(base + [f'{option}={value}'])
                        self.assertEqual(caught.exception.code, 2)
            for name in ('DEBUG_GEMM_TRAINED', 'DEBUG_GEMM', 'DEBUG_KERNEL'):
                # Even "0" enables the existing truthy-string diagnostic modes.
                with self.subTest(name=name), patch.dict('os.environ', {name: '0'}):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as caught:
                            stream.main(base)
                    self.assertEqual(caught.exception.code, 2)
        self.assertFalse(self.args.report.exists())


if __name__ == '__main__':
    unittest.main()

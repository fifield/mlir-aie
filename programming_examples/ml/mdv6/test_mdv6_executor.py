"""CPU-only lifecycle and benchmark-boundary tests; no NPU imports needed."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mdv6_executor import MDV6Executor, LegacyGraphExecutor, reject_diagnostics
from benchmark_executor import run_frames, main as benchmark_main


def backend():
    model = Mock(return_value='reference')
    return SimpleNamespace(load_model=Mock(return_value=model),
                           pad_conv0_weights=Mock(return_value=object()),
                           run_hybrid_forward=Mock(return_value=('output', 0.1)),
                           torch=SimpleNamespace(bfloat16='bf16', no_grad=contextlib.nullcontext))


def frame():
    return SimpleNamespace(shape=(1, 3, 640, 640), device=SimpleNamespace(type='cpu'),
                           dtype='bf16', is_contiguous=lambda: True)


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_persists_model_and_padded_weights(self):
        fake = backend()
        executor = MDV6Executor(fake)
        x = frame()
        for _ in range(2):
            self.assertEqual(executor.run_frame(x), 'output')
        fake.load_model.assert_called_once()
        fake.pad_conv0_weights.assert_called_once_with(executor.model)
        self.assertIs(fake.run_hybrid_forward.call_args.args[2], executor.conv0_weights)
        executor.model.assert_not_called()
        self.assertEqual(executor.reference(x), 'reference')

    def test_legacy_reconstructs_before_reference(self):
        fake = backend()
        executor = LegacyGraphExecutor(fake)
        executor.reference(frame())
        self.assertEqual(fake.load_model.call_count, 2)
        self.assertIsNone(executor.conv0_weights)

    def test_invalid_input_rejected_without_launch(self):
        fake = backend()
        executor = MDV6Executor(fake)
        x = frame()
        for attr, value in [('shape', (1, 3, 32, 32)), ('dtype', 'f32'),
                            ('device', SimpleNamespace(type='npu'))]:
            invalid = frame()
            setattr(invalid, attr, value)
            with self.assertRaises(ValueError):
                executor.run_frame(invalid)
        fake.run_hybrid_forward.assert_not_called()

    def test_failure_invalidates_executor(self):
        fake = backend()
        executor = MDV6Executor(fake)
        fake.run_hybrid_forward.side_effect = OSError('device failure')
        with self.assertRaises(OSError):
            executor.run_frame(frame())
        with self.assertRaisesRegex(RuntimeError, 'invalid'):
            executor.run_frame(frame())
        self.assertEqual(fake.run_hybrid_forward.call_count, 1)

    def test_debug_zero_is_still_enabled(self):
        with patch.dict(os.environ, {'DEBUG_GEMM': '0'}):
            with self.assertRaisesRegex(ValueError, 'DEBUG_GEMM'):
                reject_diagnostics()

    def test_benchmark_order_and_counter_deltas(self):
        events = []
        executor = SimpleNamespace(route='test', reference=lambda x: events.append('reference'),
                                   run_frame=lambda x: events.append('forward'))
        def clock():
            events.append('clock')
            return len(events)
        def compare(a, b):
            events.append('compare')
            return dict(ok=True, finite=True, max_class_diff=0.1, max_vector_diff=0.01)
        snapshots = iter([{'run_calls': 0, 'dma': None}, {'run_calls': 2, 'dma': None}])
        report = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()):
            status, rows = run_frames(executor, lambda s: events.append('input'), compare,
                                      1, 42, report, lambda: next(snapshots), clock)
        self.assertEqual(status, 0)
        self.assertEqual(events, ['input', 'reference', 'clock', 'forward', 'clock', 'compare'])
        self.assertEqual(rows[0]['runtime_delta'], {'run_calls': 2, 'dma': None})
        self.assertEqual(json.loads(report.getvalue())['seed'], 42)

    def test_first_failure_stops_and_flushes(self):
        executor = SimpleNamespace(route='test', reference=lambda x: None,
                                   run_frame=Mock(side_effect=OSError('timeout')))
        report = io.StringIO()
        snapshots = iter([{'run_calls': 0}, {'run_calls': 1}])
        with contextlib.redirect_stdout(io.StringIO()):
            status, rows = run_frames(executor, lambda s: s, Mock(), 3, 42, report,
                                      snapshot=lambda: next(snapshots))
        self.assertEqual(status, 1)
        self.assertEqual(len(rows), 1)
        self.assertIn('timeout', json.loads(report.getvalue())['error'])
        self.assertEqual(rows[0]['runtime_delta']['run_calls'], 1)

    def test_invalid_args_fail_before_executor_construction(self):
        with patch('benchmark_executor.MDV6Executor') as factory:
            for option in (['--frames', '0'], ['--class-tolerance', 'nan'],
                           ['--vector-tolerance', '-1']):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    benchmark_main(['--report', '/tmp/not-created-mdv6-test'] + option)
            factory.assert_not_called()

    def test_existing_report_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'report.jsonl'
            report.touch()
            with patch('benchmark_executor.MDV6Executor') as factory:
                with self.assertRaises(FileExistsError):
                    benchmark_main(['--report', str(report)])
                factory.assert_not_called()

    def test_changed_diagnostics_rejected_after_construction(self):
        fake = backend()
        executor = MDV6Executor(fake)
        with patch.dict(os.environ, {'DEBUG_LAYERS': '1'}):
            with self.assertRaises(ValueError):
                executor.run_frame(frame())
        fake.run_hybrid_forward.assert_not_called()


if __name__ == '__main__':
    unittest.main()

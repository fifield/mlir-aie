"""Generated-IR completion checks; requires installed AIE Python, not hardware."""

import contextlib
import importlib.util
import io
import os
import pathlib
import re
import tempfile
import unittest


try:
    from aie.iron.device import NPU2
except ImportError:
    NPU2 = None


class GemmBuildFreshnessTests(unittest.TestCase):
    def test_missing_or_stale_artifacts_rebuild(self):
        from gemm_conv1x1.build_gemm_conv1x1 import _artifacts_current
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            outputs = [root / name for name in ("model.xclbin", "model.bin", "model.mlir")]
            inputs = [root / name for name in ("kernel.o", "kernel.cc", "helper.h", "generator.py")]
            for path in outputs + inputs:
                path.touch()
                os.utime(path, (200 if path in outputs else 100,) * 2)
            self.assertTrue(_artifacts_current(outputs, inputs))
            for path in outputs + inputs:
                saved = path.stat().st_mtime
                path.unlink()
                self.assertFalse(_artifacts_current(outputs, inputs), path.name)
                path.touch()
                os.utime(path, (saved,) * 2)
            for path in inputs:
                os.utime(path, (300,) * 2)
                self.assertFalse(_artifacts_current(outputs, inputs), path.name)
                os.utime(path, (100,) * 2)


@unittest.skipIf(NPU2 is None, "installed AIE Python is required for IR checks")
class GemmCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = pathlib.Path(__file__).parent / "gemm_conv1x1/aie2_gemm_conv1x1.py"
        spec = importlib.util.spec_from_file_location("completion_generator", path)
        cls.generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.generator)

    def generate(self, **kwargs):
        with contextlib.redirect_stderr(io.StringIO()):
            return str(self.generator.gemm_conv1x1(NPU2(), **kwargs))

    def check_completions(self, text, columns, batches=1):
        outputs = re.findall(
            r"(%\w+) = aiex\.dma_configure_task_for @col_out_\w+ \{"
            r".*?\} \{issue_token = true[^}]*\}", text, re.DOTALL)
        awaits = re.findall(r"aiex\.dma_await_task\((%\w+)\)", text)
        self.assertEqual(len(outputs), columns * batches)
        self.assertEqual(awaits, outputs)
        # All output completions in a spatial group precede every free, and
        # the next group's DMA configurations come only after those frees.
        runtime = text[text.index("runtime_sequence"):]
        configurations = list(re.finditer(
            r"(%\w+) = aiex\.dma_configure_task_for @(\w+)", runtime))
        config_cursor = 0
        for group in range(batches):
            group_outputs = outputs[group * columns:(group + 1) * columns]
            last_await = runtime.index(f"aiex.dma_await_task({group_outputs[-1]})")
            group_start = configurations[config_cursor].start()
            group_configs = []
            while config_cursor < len(configurations):
                config = configurations[config_cursor]
                group_configs.append(config)
                config_cursor += 1
                if config.group(1) == group_outputs[-1]:
                    break
            self.assertEqual(group_configs[-1].group(1), group_outputs[-1])
            self.assertNotIn("aiex.dma_free_task", runtime[group_start:last_await])
            free_positions = []
            for config in group_configs:
                if config.group(1) in group_outputs:
                    continue  # Awaited output descriptors are freed by await.
                free = f"aiex.dma_free_task({config.group(1)})"
                self.assertEqual(runtime.count(free), 1)
                free_positions.append(runtime.index(free))
            self.assertTrue(free_positions)
            self.assertGreater(min(free_positions), last_await)
            if config_cursor < len(configurations):
                self.assertGreater(configurations[config_cursor].start(), max(free_positions))
        self.assertEqual(config_cursor, len(configurations))

    def test_default_kblocked_and_nonblocked(self):
        for ic, oc, kb, tile_m, ppc in ((256, 128, 16, 68, 1),
                                       (128, 64, 0, 104, 2)):
            with self.subTest(k_block=kb):
                self.check_completions(self.generate(
                    ic=ic, oc=oc, k_block=kb, tile_m=tile_m,
                    patches_per_core=ppc, n_cores=32), 8)

    def test_partial_column_and_single_core(self):
        for cores, columns in ((5, 2), (1, 1)):
            with self.subTest(cores=cores):
                self.check_completions(self.generate(
                    ic=16, oc=16, tile_m=8, n_cores=cores), columns)

    def test_bounded_spatial_groups(self):
        for ic, oc, kb, tile_m, ppc, batches in ((256, 128, 16, 68, 1, 3),
                                                (128, 64, 0, 104, 2, 4)):
            with self.subTest(k_block=kb):
                self.check_completions(self.generate(
                    ic=ic, oc=oc, k_block=kb, tile_m=tile_m,
                    patches_per_core=ppc, n_cores=32,
                    spatial_batches=batches), 8, batches)

    def test_unsafe_completion_rejected(self):
        with self.assertRaisesRegex(ValueError, "partial completion is unsafe"):
            self.generate(wait_all_columns=False)

    def test_schedule_checker_rejects_early_free_and_next_group(self):
        text = self.generate(ic=16, oc=16, tile_m=8, n_cores=1,
                             spatial_batches=2)
        first_free = re.search(r"aiex\.dma_free_task\(%\w+\)", text).group(0)
        first_await = text.index("aiex.dma_await_task")
        early_free = text.replace(first_free, "", 1)
        early_free = early_free[:first_await] + first_free + "\n" + early_free[first_await:]
        with self.assertRaises(AssertionError):
            self.check_completions(early_free, 1, 2)
        # Delay a first-group free until after the next group has configured.
        delayed_free = text.replace(first_free, "", 1)
        next_config = delayed_free.index("aiex.dma_configure_task_for", first_await)
        line_end = delayed_free.index("\n", next_config)
        delayed_free = delayed_free[:line_end] + "\n" + first_free + delayed_free[line_end:]
        with self.assertRaises(AssertionError):
            self.check_completions(delayed_free, 1, 2)

    def test_cli_default_and_compatibility_alias_are_safe(self):
        self.assertTrue(self.generator._parse_args([]).wait_all_columns)
        self.assertTrue(self.generator._parse_args(["--wait-all-columns"]).wait_all_columns)


if __name__ == "__main__":
    if NPU2 is None:
        raise RuntimeError("installed AIE Python is required for the generated-IR gate")
    unittest.main()

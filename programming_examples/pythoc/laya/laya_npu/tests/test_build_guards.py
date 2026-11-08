"""Builders refuse to overwrite an existing output directory."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BuildGuards(unittest.TestCase):
    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "existing"
            out.mkdir()
            sentinel = out / "evidence"
            sentinel.write_text("preserve")
            run = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "build.py"),
                    "--target",
                    "residual",
                    "--output",
                    str(out),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("output must not exist", run.stderr)
            self.assertEqual(sentinel.read_text(), "preserve")
            self.assertEqual(list(out.iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()

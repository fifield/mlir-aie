# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise missing-wrapper diagnostics through an isolated aiecc install."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

design = Path(sys.argv[1]).resolve()
prefix = Path(sys.argv[2]).resolve()
(prefix / "bin").mkdir(parents=True, exist_ok=True)
(prefix / "aie_runtime_lib").mkdir(exist_ok=True)
(prefix / "no-aietools").mkdir(exist_ok=True)
exe = prefix / "bin" / ("aiecc.exe" if os.name == "nt" else "aiecc")
source = shutil.which("aiecc")
assert source, "aiecc must be on PATH"
if exe.exists():
    exe.unlink()
try:
    os.link(source, exe)
except OSError:
    shutil.copy2(source, exe)
command = [
    str(exe),
    str(design),
    "--xchesscc",
    "--get=chesslinked_{0}.ll",
    f"--aietools={prefix / 'no-aietools'}",
    f"--output-dir={prefix / 'out'}",
    f"--tmpdir={prefix / 'work'}",
    "-v",
]
result = subprocess.run(command, capture_output=True, text=True)
assert result.returncode != 0, result.stdout + result.stderr
assert "missing Chess intrinsic wrapper:" in result.stderr, result.stderr
assert "AIE4/chess_intrinsic_wrapper.ll" in result.stderr.replace("\\", "/")
assert "AIE4 enabled in AIE_VITIS_COMPONENTS" in result.stderr, result.stderr
assert "exec:" not in result.stderr, result.stderr

dry_run = subprocess.run([*command, "-n"], capture_output=True, text=True)
assert dry_run.returncode == 0, dry_run.stdout + dry_run.stderr
assert "chess-llvm-link" in dry_run.stdout + dry_run.stderr
assert "missing Chess intrinsic wrapper" not in dry_run.stderr

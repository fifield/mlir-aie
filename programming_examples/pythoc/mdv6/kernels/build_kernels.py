#!/usr/bin/env python3
# build_kernels.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

"""Inline-compile the 4 rep_elan_bf16 kernels to one .o per kernel.

Produces:
  build/conv3x3_fused_packed_bf16.o
  build/gemm_conv1x1_fused_packed_bf16.o
  build/gemm_conv1x1_kblocked_bf16.o
  build/residual_add_silu_bf16.o

This is now a thin wrapper around ``rep_elan_bf16_pythoc.build_all_objs``:
the four kernels are real PythoC ``@aie_kernel`` functions, not external
C++ TUs. Each call to a ``make_*`` factory inline-compiles its kernel to
LLVM IR and produces a ``.o``; we copy each one to its canonical
``build/<name>.o`` filename so IRON wrappers that pass
``Kernel(name, "<name>.o", ...)`` continue to work.

Usage
-----
  source /home/jfifield/npu-dev-pythoc/env.sh
  python mlir-aie/programming_examples/pythoc/mdv6/kernels/build_kernels.py
  python mlir-aie/programming_examples/pythoc/mdv6/kernels/build_kernels.py --clean
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD_DIR = HERE / "build"

# Importing rep_elan_bf16_pythoc installs the subprocess.run patch that
# disables opt's loop vectorizer for the bf16 BN+SiLU tail.
sys.path.insert(0, str(HERE))
import rep_elan_bf16_pythoc as _mod  # noqa: E402


KERNELS = (
    "conv3x3_fused_packed_bf16",
    "gemm_conv1x1_fused_packed_bf16",
    "gemm_conv1x1_kblocked_bf16",
    "residual_add_silu_bf16",
)


def clean() -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Inline-compile the 4 rep_elan_bf16 PythoC kernels to .o files."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the build directory and exit.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.clean:
        clean()
        print("[build_kernels] cleaned.")
        return 0

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    outs = _mod.build_all_objs(BUILD_DIR)
    for o in outs:
        print(f"  -> {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

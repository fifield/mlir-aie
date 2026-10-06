#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Require translated calls to cover the generated AIE4 XLLVM intrinsic names."""

import argparse
import re
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("conversions", type=Path)
    ap.add_argument("llvm_ir", type=Path, nargs="+")
    args = ap.parse_args()

    expected = set(
        re.findall(r'"(llvm\.aie4\.[A-Za-z0-9_.]+)"', args.conversions.read_text())
    )
    if not expected:
        sys.exit("error: no AIE4 intrinsic names in generated XLLVM conversions")

    # Count calls, not declarations: an unused declaration exercises no verifier.
    called = set()
    for path in args.llvm_ir:
        called.update(
            re.findall(
                r'\bcall\b[^@\n]*@"?(llvm\.aie4\.[A-Za-z0-9_.]+)"?\(', path.read_text()
            )
        )
    missing, unexpected = expected - called, called - expected
    for name in sorted(missing):
        print(f"MISSING XLLVM CALL: {name}")
    for name in sorted(unexpected):
        print(f"UNEXPECTED XLLVM CALL: {name}")
    if missing or unexpected:
        sys.exit(1)
    print(
        f"PASS: all {len(expected)} generated AIE4 XLLVM intrinsic names have translated calls"
    )


if __name__ == "__main__":
    main()

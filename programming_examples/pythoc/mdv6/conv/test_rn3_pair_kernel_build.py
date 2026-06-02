#!/usr/bin/env python3
"""Build smoke for the rn3 pair fused PythoC kernel object."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels"))

from rn3_pair_pythoc import build_obj  # noqa: E402


def test_rn3_pair_kernel_object_builds():
    out = build_obj(ROOT / "kernels" / "build")
    assert out.name == "rn3_pair_fused_bf16.o"
    assert out.exists()
    assert out.stat().st_size > 0


if __name__ == "__main__":
    test_rn3_pair_kernel_object_builds()
    print("PASS: rn3 pair fused kernel object builds")

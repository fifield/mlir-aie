#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Opt-in NPU smoke tests for the packed-AWQ GEMV primitive.

These tests compile and execute tiny AWQ GEMV kernels on the NPU. They are
skipped by default so normal unit-test runs do not require hardware.
Set RUN_AWQ_NPU_SMOKE=1 to enable.

Stage-1 note: the K=8 / K=16, M=4, group=4 shapes below are intentionally NOT
in the awq_impl cached-MLIR set (which only ships the model shapes
k=2048 m=32 g=128 and k=8192 m=8 g=128). They are kept here as the original
hand-checked correctness cases because the C++-compile path in
`kernel_builder/external_kernels.py::compile_awq_gemv` accepts any (K, M, G);
the runtime will compile-on-demand via `awq_gemv_builder.build_awq_gemv_ir`
(aircc lowering) on first call. Stage 2 retires the C++/AIR path; at that
point either the smoke shapes must be retargeted to a representative model
shape, or PythoC dim-specialized clones for these tiny shapes must be added.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from ml_dtypes import bfloat16

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXAMPLE_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_EXAMPLE_DIR))

from kernel_builder.cache import KernelCache  # noqa: E402
from llama32_1b_awq_runtime import awq_gemv_npu  # noqa: E402
from llama32_1b_weights import AwqLinear, awq_gemv_cpu_reference  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_AWQ_NPU_SMOKE") != "1",
    reason="set RUN_AWQ_NPU_SMOKE=1 to compile/run tiny AWQ GEMV kernels on NPU",
)


def _pack_uint4_rows(unpacked: np.ndarray) -> np.ndarray:
    return np.bitwise_or(
        unpacked[:, 0::2], np.left_shift(unpacked[:, 1::2], np.uint8(4))
    ).astype(np.uint8)


def _case_k8_m4_g4():
    k, m, group_size = 8, 4, 4
    q_unpacked = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [7, 6, 5, 4, 3, 2, 1, 0],
            [1, 3, 5, 7, 9, 11, 13, 15],
            [15, 13, 11, 9, 7, 5, 3, 1],
        ],
        dtype=np.uint8,
    )
    params = np.array(
        [
            [0.5, 1, 0.25, 2],
            [1.0, 4, 0.5, 2],
            [0.125, 0, 2.0, 12],
            [0.25, 8, 1.0, 4],
        ],
        dtype=bfloat16,
    )
    x = np.array([1, -2, 0.5, 3, -1, 2, -0.5, 4], dtype=bfloat16)
    return k, m, group_size, x, _pack_uint4_rows(q_unpacked), params


def _case_k16_m4_g4():
    k, m, group_size = 16, 4, 4
    q_unpacked = np.bitwise_and(
        np.arange(m * k, dtype=np.uint8).reshape(m, k) * 3 + 1, 15
    )
    params = np.empty((m, 2 * (k // group_size)), dtype=bfloat16)
    for row in range(m):
        for group in range(k // group_size):
            params[row, 2 * group] = bfloat16(0.25 * (group + 1))
            params[row, 2 * group + 1] = bfloat16((row + group) % 5)
    x = np.array(
        [1, -2, 0.5, 3, -1, 2, -0.5, 4, 1.5, -1.5, 0.25, -0.25, 2.5, -3, 3.5, -4],
        dtype=bfloat16,
    )
    return k, m, group_size, x, _pack_uint4_rows(q_unpacked), params


@pytest.mark.parametrize("case", [_case_k8_m4_g4, _case_k16_m4_g4])
def test_tiny_awq_gemv_npu_matches_cpu_reference(case, tmp_path, monkeypatch):
    k, m, group_size, x, qweight, params = case()
    monkeypatch.chdir(tmp_path)

    cache = KernelCache(cache_dir=tmp_path / "awq_gemv_kernel_cache", verbose=False)
    awq = AwqLinear(qweight=qweight, params=params, k=k, m=m, group_size=group_size)
    out = awq_gemv_npu(cache, x, awq).astype(np.float32)

    expected = awq_gemv_cpu_reference(x, awq, dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=0.125)

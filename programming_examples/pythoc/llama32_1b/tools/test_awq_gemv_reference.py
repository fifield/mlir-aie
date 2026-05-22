#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Independent tests for direct repacked-AWQ GEMV consumption.

These tests intentionally use an explicit scalar loop as the oracle instead of
_dequant_repacked_awq_linear(), so nibble order, group indexing, and params
layout are locked down before an AIE kernel consumes the same tensors.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_SCRIPT_DIR = os.path.dirname(__file__)
_EXAMPLE_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _EXAMPLE_DIR)

from llama32_1b_weights import (  # noqa: E402
    AwqLinear,
    _dequant_repacked_awq_linear,
    awq_gemv_cpu_reference,
)


def _pack_row(nibbles: list[int]) -> list[int]:
    assert len(nibbles) % 2 == 0
    return [int(nibbles[i]) | (int(nibbles[i + 1]) << 4) for i in range(0, len(nibbles), 2)]


def _scalar_awq_gemv(x: np.ndarray, awq: AwqLinear) -> np.ndarray:
    x32 = np.asarray(x, dtype=np.float32)
    params = np.asarray(awq.params, dtype=np.float32)
    out = np.zeros(awq.m, dtype=np.float32)
    for m in range(awq.m):
        acc = np.float32(0.0)
        for k in range(awq.k):
            packed = int(awq.qweight[m, k // 2])
            q = (packed & 0xF) if k % 2 == 0 else ((packed >> 4) & 0xF)
            group = k // awq.group_size
            scale = params[m, 2 * group]
            zero = params[m, 2 * group + 1]
            acc += x32[k] * ((np.float32(q) - zero) * scale)
        out[m] = acc
    return out


def test_awq_gemv_scalar_oracle_locks_down_nibble_group_and_param_layout():
    # M=3, K=8, group_size=4 gives two scale/zero pairs per row. Row 0 is
    # ascending so low/high nibble swaps are obvious. Other rows exercise zeros
    # and different scales across groups.
    qweight = np.array(
        [
            _pack_row([0, 1, 2, 3, 4, 5, 6, 7]),
            _pack_row([15, 14, 13, 12, 11, 10, 9, 8]),
            _pack_row([1, 3, 5, 7, 9, 11, 13, 15]),
        ],
        dtype=np.uint8,
    )
    params = np.array(
        [
            [0.5, 1.0, 0.25, 2.0],
            [1.0, 8.0, 0.5, 10.0],
            [0.125, 0.0, 2.0, 12.0],
        ],
        dtype=bfloat16,
    )
    x = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 2.0, -0.5, 4.0], dtype=bfloat16)
    awq = AwqLinear(qweight=qweight, params=params, k=8, m=3, group_size=4)

    expected = _scalar_awq_gemv(x, awq)
    got = awq_gemv_cpu_reference(x, awq, dtype=np.float32)

    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


def test_awq_gemv_cpu_reference_matches_dequantized_matmul_without_expanding_in_api():
    rng = np.random.default_rng(20260521)
    k = 16
    m = 4
    group_size = 4
    q_unpacked = rng.integers(0, 16, size=(m, k), dtype=np.uint8)
    qweight = (q_unpacked[:, 0::2] | (q_unpacked[:, 1::2] << np.uint8(4))).astype(np.uint8)
    groups = k // group_size
    scales = rng.uniform(0.01, 0.2, size=(m, groups)).astype(np.float32)
    zeros = rng.integers(0, 16, size=(m, groups)).astype(np.float32)
    params = np.empty((m, groups * 2), dtype=bfloat16)
    params[:, 0::2] = scales.astype(bfloat16)
    params[:, 1::2] = zeros.astype(bfloat16)
    x = rng.normal(size=k).astype(bfloat16)
    awq = AwqLinear(qweight=qweight, params=params, k=k, m=m, group_size=group_size)

    got = awq_gemv_cpu_reference(x, awq, dtype=np.float32)
    deq_rows = _dequant_repacked_awq_linear(awq, dtype=np.float32)
    expected = np.asarray(x, dtype=np.float32) @ deq_rows.T

    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-5)


def test_awq_gemv_cpu_reference_rejects_wrong_input_length():
    awq = AwqLinear(
        qweight=np.zeros((1, 4), dtype=np.uint8),
        params=np.ones((1, 2), dtype=bfloat16),
        k=8,
        m=1,
        group_size=8,
    )
    try:
        awq_gemv_cpu_reference(np.ones(7, dtype=bfloat16), awq)
    except ValueError as exc:
        assert "input length 7 does not match AWQ K=8" in str(exc)
    else:  # pragma: no cover - clearer assertion than pytest.raises import here
        raise AssertionError("expected ValueError for wrong input length")


def main() -> int:
    test_awq_gemv_scalar_oracle_locks_down_nibble_group_and_param_layout()
    print("PASS test_awq_gemv_scalar_oracle_locks_down_nibble_group_and_param_layout")
    test_awq_gemv_cpu_reference_matches_dequantized_matmul_without_expanding_in_api()
    print("PASS test_awq_gemv_cpu_reference_matches_dequantized_matmul_without_expanding_in_api")
    test_awq_gemv_cpu_reference_rejects_wrong_input_length()
    print("PASS test_awq_gemv_cpu_reference_rejects_wrong_input_length")
    print("PASS test_awq_gemv_reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

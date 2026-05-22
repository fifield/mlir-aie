#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for opt-in packed-AWQ GEMV runtime wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXAMPLE_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_EXAMPLE_DIR))

import llama32_1b_awq_runtime as awq_runtime  # noqa: E402
from llama32_1b_weights import AwqLinear  # noqa: E402


class _FakeCache:
    def __init__(self):
        self.compile_calls = []
        self.run_calls = []
        self.artifacts = {}

    def compile_and_cache(self, name, ir, instance_name):
        self.compile_calls.append((name, ir, instance_name))
        self.artifacts[name] = object()

    def load_and_run(self, name, backend, *inputs, output_indices):
        self.run_calls.append((name, backend, inputs, output_indices))
        y = np.asarray(inputs[3], dtype=bfloat16).copy()
        y[:] = np.arange(y.shape[0], dtype=np.float32).astype(bfloat16)
        return {3: y}


def _tiny_awq():
    qweight = np.array([[0x10, 0x32], [0x54, 0x76]], dtype=np.uint8)
    params = np.array([[0.5, 1, 0.25, 2], [1.0, 4, 0.5, 2]], dtype=bfloat16)
    return AwqLinear(qweight=qweight, params=params, k=4, m=2, group_size=2)


def test_awq_gemv_npu_wrapper_uses_direct_awq_buffers_and_distinct_kernel(monkeypatch):
    monkeypatch.setattr(awq_runtime, "build_awq_gemv_ir", lambda k, m, group_size, **kwargs: "mock-ir")
    cache = _FakeCache()
    awq = _tiny_awq()
    x = np.array([1, -2, 0.5, 3], dtype=bfloat16)

    out = awq_runtime.awq_gemv_npu(cache, x, awq)

    assert out.dtype == bfloat16
    np.testing.assert_array_equal(out.astype(np.float32), np.array([0, 1], dtype=np.float32))
    assert cache.compile_calls == [("awq_gemv_k4_m2_g2_scalar", "mock-ir", "awq_gemv")]
    name, _backend, inputs, output_indices = cache.run_calls[0]
    assert name == "awq_gemv_k4_m2_g2_scalar"
    assert output_indices == [3]
    np.testing.assert_array_equal(inputs[1], awq.qweight.reshape(-1))
    np.testing.assert_array_equal(inputs[2], awq.params.reshape(-1))



def test_awq_gemv_npu_wrapper_vecdeq_uses_distinct_kernel(monkeypatch):
    seen = []

    def fake_build(k, m, group_size, **kwargs):
        seen.append((k, m, group_size, kwargs))
        return "mock-ir"

    monkeypatch.setattr(awq_runtime, "build_awq_gemv_ir", fake_build)
    cache = _FakeCache()
    awq = _tiny_awq()
    x = np.array([1, -2, 0.5, 3], dtype=bfloat16)

    awq_runtime.awq_gemv_npu(cache, x, awq, variant="vecdeq")

    assert seen == [(4, 2, 2, {"variant": "vecdeq"})]
    assert cache.compile_calls == [("awq_gemv_k4_m2_g2_vecdeq", "mock-ir", "awq_gemv")]
    assert cache.run_calls[0][0] == "awq_gemv_k4_m2_g2_vecdeq"

def test_awq_gemv_npu_wrapper_rejects_bad_input_length(monkeypatch):
    monkeypatch.setattr(awq_runtime, "build_awq_gemv_ir", lambda k, m, group_size, **kwargs: "mock-ir")
    try:
        awq_runtime.awq_gemv_npu(_FakeCache(), np.zeros(3, dtype=bfloat16), _tiny_awq())
    except ValueError as exc:
        assert "does not match AWQ K" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected bad input length to fail")


def test_awq_gemv_npu_tiled_runs_full_linear_in_row_chunks(monkeypatch):
    monkeypatch.setattr(awq_runtime, "build_awq_gemv_ir", lambda k, m, group_size, **kwargs: "mock-ir")
    cache = _FakeCache()
    qweight = np.arange(6 * 2, dtype=np.uint8).reshape(6, 2)
    params = np.arange(6 * 4, dtype=np.float32).reshape(6, 4).astype(bfloat16)
    awq = AwqLinear(qweight=qweight, params=params, k=4, m=6, group_size=2)
    x = np.array([1, -2, 0.5, 3], dtype=bfloat16)

    out = awq_runtime.awq_gemv_npu_tiled(cache, x, awq, tile_m=2)

    assert out.dtype == bfloat16
    np.testing.assert_array_equal(out.astype(np.float32), np.array([0, 1, 0, 1, 0, 1], dtype=np.float32))
    assert [call[0] for call in cache.compile_calls] == ["awq_gemv_k4_m2_g2_scalar"]
    assert len(cache.run_calls) == 3
    for tile_idx, (_name, _backend, inputs, output_indices) in enumerate(cache.run_calls):
        assert output_indices == [3]
        row0 = tile_idx * 2
        np.testing.assert_array_equal(inputs[1], qweight[row0 : row0 + 2].reshape(-1))
        np.testing.assert_array_equal(inputs[2], params[row0 : row0 + 2].reshape(-1))


def test_awq_gemv_npu_tiled_handles_partial_final_tile(monkeypatch):
    monkeypatch.setattr(awq_runtime, "build_awq_gemv_ir", lambda k, m, group_size, **kwargs: "mock-ir")
    cache = _FakeCache()
    qweight = np.arange(5 * 2, dtype=np.uint8).reshape(5, 2)
    params = np.arange(5 * 4, dtype=np.float32).reshape(5, 4).astype(bfloat16)
    awq = AwqLinear(qweight=qweight, params=params, k=4, m=5, group_size=2)
    x = np.array([1, -2, 0.5, 3], dtype=bfloat16)

    out = awq_runtime.awq_gemv_npu_tiled(cache, x, awq, tile_m=2)

    np.testing.assert_array_equal(out.astype(np.float32), np.array([0, 1, 0, 1, 0], dtype=np.float32))
    assert [call[0] for call in cache.compile_calls] == [
        "awq_gemv_k4_m2_g2_scalar",
        "awq_gemv_k4_m1_g2_scalar",
    ]
    assert cache.run_calls[-1][2][1].shape == (2,)
    assert cache.run_calls[-1][2][2].shape == (4,)

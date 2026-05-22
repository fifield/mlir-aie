#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for opt-in experimental packed-AWQ decode wiring."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ml_dtypes import bfloat16

import llama32_1b_awq_runtime as awq_runtime
from llama32_1b_decode import _run_o_ffn_awq_experimental
from llama32_1b_reference import rms_norm


def _awq(name: str, k: int, m: int):
    return SimpleNamespace(name=name, k=k, m=m)


def test_o_ffn_awq_experimental_runs_four_packed_awq_npu_gemvs(monkeypatch):
    calls = []

    def fake_awq_gemv_npu_tiled(cache, x, awq, *, tile_m, variant="scalar"):
        calls.append((awq.name, np.asarray(x).shape, tile_m, variant))
        values = {
            "wo": np.array([0.25, -0.5, 0.75, -1.0], dtype=np.float32),
            "gate": np.linspace(-0.3, 0.2, 6, dtype=np.float32),
            "up": np.linspace(0.5, 1.0, 6, dtype=np.float32),
            "down": np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
        }[awq.name]
        return values.astype(bfloat16)

    monkeypatch.setattr(awq_runtime, "awq_gemv_npu_tiled", fake_awq_gemv_npu_tiled)

    cache = object()
    attn_out = np.array([1.0, 2.0, 3.0, 4.0], dtype=bfloat16)
    x_residual = np.array([0.5, -0.25, 0.125, -0.0625], dtype=bfloat16)
    layer_weights = SimpleNamespace(ffn_norm=np.ones(4, dtype=bfloat16))
    awq_layer = SimpleNamespace(
        wo=_awq("wo", k=2048, m=4),
        w_gate=_awq("gate", k=2048, m=6),
        w_up=_awq("up", k=2048, m=6),
        w_down=_awq("down", k=8192, m=4),
    )

    out = _run_o_ffn_awq_experimental(
        cache,
        attn_out,
        x_residual,
        layer_weights,
        awq_layer,
        emb_dim=4,
        awq_tile_m_k2048=32,
        awq_tile_m_k8192=8,
    )

    assert calls == [
        ("wo", (4,), 32, "vecdeq"),
        ("gate", (4,), 32, "vecdeq"),
        ("up", (4,), 32, "vecdeq"),
        ("down", (6,), 8, "vecdeq"),
    ]

    proj = np.array([0.25, -0.5, 0.75, -1.0], dtype=np.float32)
    res1 = (proj + x_residual.astype(np.float32)).astype(bfloat16)
    normed2 = rms_norm(res1.astype(np.float32).reshape(1, 4), np.ones(4, dtype=np.float32))
    # The fake asserts the down input length via calls. Its returned down vector
    # is then residual-added by the experimental path.
    expected = (
        np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        + res1.astype(np.float32)
    ).astype(bfloat16)
    assert normed2.shape == (1, 4)
    np.testing.assert_array_equal(out, expected)

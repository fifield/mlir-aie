#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke tests for packed-AWQ matvec and fused O+FFN plumbing."""

from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(__file__)
_EXAMPLE_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _EXAMPLE_DIR)


def test_awq_matvec_builder_matches_bf16_gemv_tiled_abi_shape():
    from kernel_builder.awq_matvec import build_module

    text = str(build_module(m=128, k=128, group_size=32, tile_m=8, m_input=4, herd_m=8))
    assert "func.func @awq_matvec" in text
    assert "awq_matvec_vectorized_u4_bf16" in text
    assert "awq_linalg_fill_bf16" in text
    assert 'link_with = "awq_mv.o"' in text
    assert "memref<128x64xui8>" in text
    assert "memref<128x8xbf16>" in text
    assert "memref<128xbf16>" in text
    assert text.count("air.launch") == 1
    assert text.count("air.herd") == 1


def test_fused_awq_o_gemv_ffn_uses_one_public_func_and_eight_launches():
    from kernel_builder.o_gemv_ffn_awq_stitched import build_o_gemv_ffn_awq_module

    text = str(
        build_o_gemv_ffn_awq_module(
            emb_dim=128,
            hidden_dim=512,
            group_size=32,
            tile_m=8,
            m_input=4,
            down_tile_m=2,
            down_m_input=1,
            herd_m=8,
        )
    )
    assert "func.func @o_gemv_ffn_awq" in text
    assert text.count("air.launch") == 8
    assert 'link_with = "awq_mv.o"' in text
    assert 'link_with = "awq_mv_k8192.o"' in text
    assert "dg_awq_matvec_vectorized_u4_bf16" in text
    assert "memref<128x64xui8>" in text
    assert "memref<512x64xui8>" in text
    assert "memref<128x256xui8>" in text
    assert "memref<128x32xbf16>" in text


def test_awq_o_gemv_ffn_backend_name_is_distinct():
    from kernel_builder.backend_presets import OGF_AWQ_BACKEND

    assert OGF_AWQ_BACKEND["instance_name"] == "o_gemv_ffn_awq"
    assert OGF_AWQ_BACKEND["instance_name"] != "o_gemv_ffn"


def test_awq_mv_external_compile_helpers_use_distinct_objects(monkeypatch):
    from kernel_builder import external_kernels

    calls = []

    def fake_compile(src_path, output_name, extra_flags=None, force=False):
        calls.append((str(src_path), output_name, list(extra_flags or []), force))

    monkeypatch.setattr(external_kernels, "_compile_kernel", fake_compile)
    external_kernels.compile_awq_mv(group_size=128, tile_m=8)
    external_kernels.compile_awq_mv_k8192(group_size=128, tile_m=2)

    assert calls[0][1] == "awq_mv.o"
    assert "-DDIM_M_OUTPUT=8" in calls[0][2]
    assert calls[1][1] == "awq_mv_k8192.o"
    assert "-DDIM_M_OUTPUT=2" in calls[1][2]
    assert "-DAWQ_MATVEC_FN=dg_awq_matvec_vectorized_u4_bf16" in calls[1][2]
    assert "-DAWQ_LINALG_FILL_FN=dg_awq_linalg_fill_bf16" in calls[1][2]


def test_fused_awq_runtime_uses_single_xrt_call_with_packed_args():
    import numpy as np
    from ml_dtypes import bfloat16
    from types import SimpleNamespace
    from llama32_1b_awq_runtime import o_gemv_ffn_awq_npu
    from llama32_1b_weights import AwqLinear

    emb_dim = 128
    hidden_dim = 512
    group_size = 32

    def awq(m, k):
        return AwqLinear(
            qweight=np.zeros((m, k // 2), dtype=np.uint8),
            params=np.zeros((m, 2 * (k // group_size)), dtype=bfloat16),
            k=k,
            m=m,
            group_size=group_size,
        )

    awq_layer = SimpleNamespace(
        wo=awq(emb_dim, emb_dim),
        w_gate=awq(hidden_dim, emb_dim),
        w_up=awq(hidden_dim, emb_dim),
        w_down=awq(emb_dim, hidden_dim),
    )

    class FakeCache:
        artifacts = {"o_gemv_ffn_awq": object()}

        def __init__(self):
            self.calls = []

        def load_and_run(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return tuple(
                np.zeros_like(a) if isinstance(a, np.ndarray) else np.empty(0, dtype=bfloat16)
                for a in args[2:]
            )

    cache = FakeCache()
    out = o_gemv_ffn_awq_npu(
        cache,
        np.zeros((emb_dim,), dtype=bfloat16),
        np.zeros((emb_dim,), dtype=bfloat16),
        np.ones((emb_dim,), dtype=bfloat16),
        awq_layer,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        layer_idx=3,
    )

    assert out.shape == (emb_dim,)
    assert len(cache.calls) == 1
    args, kwargs = cache.calls[0]
    assert args[0] == "o_gemv_ffn_awq"
    # Current AWQ runtime ABI (llama32_1b_awq_runtime.py:201-223): 15 positional
    # args after (name, backend) -- one combined uint8 weight buffer per AWQ
    # GEMV (wo_w, wgate_w, wup_w, wdown_w) instead of the earlier separate
    # (qweight, params) pair.  output_indices=[14] (was [18] under the 19-arg ABI).
    assert len(args[2:]) == 15
    # arg index 2 is wo_w (combined uint8 weight buffer: K/2 qbytes + 4*groups param bytes)
    expected_wo_cols = emb_dim // 2 + 4 * (emb_dim // group_size)
    expected_wdown_cols = hidden_dim // 2 + 4 * (hidden_dim // group_size)
    assert args[2].shape == (emb_dim, expected_wo_cols)
    # arg index 14 is wdown_w (combined uint8 weight for the down projection)
    assert args[14].shape == (emb_dim, expected_wdown_cols)
    assert kwargs["output_indices"] == [14]
    assert kwargs["bo_key"] == "o_gemv_ffn_awq_L3"

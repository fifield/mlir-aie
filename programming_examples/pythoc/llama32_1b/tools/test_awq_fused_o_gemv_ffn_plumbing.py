#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke tests for packed-AWQ fused O+FFN plumbing.

After Phase 6 Stage 4, the AIR-tree builders (`kernel_builder.awq_matvec`,
`kernel_builder.o_gemv_ffn_awq_stitched`) and the temporary Peano-clang
`.cc` compile machinery are deleted. These tests now verify only the
remaining stable surfaces: backend-preset naming, placed-IRON builder
output, and runtime ABI shape.

End-to-end correctness is exercised by `make hf-gate QUANT=awq`.
"""

from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(__file__)
_EXAMPLE_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _EXAMPLE_DIR)


def test_placed_o_gemv_ffn_awq_emits_correct_link_with_and_external_funcs():
    """Phase 6 / Stage 3: placed-IRON builder emits the fused module
    with the right link_with and external-func references for both the
    K=2048 (og/gg/ug) and K=8192 (dg) AWQ matvec kernels.
    """
    from builders.o_gemv_ffn_awq import build_o_gemv_ffn_awq_module

    text = build_o_gemv_ffn_awq_module(emb_dim=2048, hidden_dim=8192)
    # External function declarations for the AWQ kernels.
    assert "awq_matvec_vectorized_u4_bf16" in text
    assert "awq_linalg_fill_bf16" in text
    assert "dg_awq_matvec_vectorized_u4_bf16" in text
    assert "dg_awq_linalg_fill_bf16" in text
    # link_with on the four GEMV devices points at the PythoC `.o` outputs.
    assert 'link_with = "awq_mv_pythoc.o"' in text
    assert 'link_with = "awq_mv_k8192_pythoc.o"' in text
    # Combined-row ABI memref shapes (group_size=128 baked).
    # emb_dim=2048: K/2 + 4*(K/group_size) = 1024 + 64 = 1088
    # hidden_dim=8192: 4096 + 256 = 4352
    assert "memref<2048x1088xui8>" in text   # wo_w (og)
    assert "memref<8192x1088xui8>" in text   # wgate_w / wup_w (gg, ug)
    assert "memref<2048x4352xui8>" in text   # wdown_w (dg)


def test_awq_o_gemv_ffn_backend_name_is_distinct():
    from kernel_builder.backend_presets import OGF_AWQ_BACKEND

    assert OGF_AWQ_BACKEND["instance_name"] == "o_gemv_ffn_awq"
    assert OGF_AWQ_BACKEND["instance_name"] != "o_gemv_ffn"


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

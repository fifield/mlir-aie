#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke tests for AWQ GEMV kernel naming/plumbing."""

from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(__file__)
_EXAMPLE_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _EXAMPLE_DIR)

from kernel_builder.external_kernels import awq_gemv_kernel_name  # noqa: E402
from kernel_builder.backend_presets import AWQ_GEMV_BACKEND  # noqa: E402


def test_awq_gemv_kernel_name_is_dimension_specialized_and_not_bf16_name():
    # Phase 6 cleanup: only `vecdeq` variant is supported after Stage 4
    # (the AIR-tree scalar builder was deleted).
    name = awq_gemv_kernel_name(k=2048, m=2048, group_size=128)
    assert name == "awq_gemv_k2048_m2048_g128_vecdeq"
    assert name not in {"rms_gemv_rope", "o_gemv_ffn", "lm_head_gemv"}
    assert awq_gemv_kernel_name(k=2048, m=2048, group_size=128, variant="vecdeq") == "awq_gemv_k2048_m2048_g128_vecdeq"


def test_awq_backend_instance_is_distinct_from_existing_bf16_gemvs():
    assert AWQ_GEMV_BACKEND["instance_name"] == "awq_gemv"
    assert AWQ_GEMV_BACKEND["instance_name"] not in {
        "rms_gemv_rope",
        "o_gemv_ffn",
        "lm_head_gemv",
    }


def test_placed_awq_matvec_links_awq_pythoc_object():
    """Phase 6 / Stage 3: the placed-IRON builder emits aie/aiex MLIR that
    references the dim-specialized PythoC kernel object.

    Replaces the pre-cleanup `build_awq_gemv_air_module` smoke (AIR-tree
    builder deleted in Stage 4).
    """
    from builders.awq_matvec import build_awq_matvec_module

    # Use one of the two ported shapes (matches reference_mlir/awq_gemv_*_pythoc.npu.air.mlir).
    text = build_awq_matvec_module(k=2048, m=32, group_size=128, variant="vecdeq")
    assert "awq_gemv_u4_bf16" in text
    assert 'link_with = "awq_gemv_k2048_m32_g128_vecdeq_pythoc.o"' in text
    assert "memref<2048xbf16>" in text       # x:       bf16[K]
    assert "memref<32768xui8>" in text       # qweight: ui8[M*(K/2)] = 32*1024
    assert "memref<32xbf16>" in text         # y:       bf16[M]


def main() -> int:
    test_awq_gemv_kernel_name_is_dimension_specialized_and_not_bf16_name()
    print("PASS test_awq_gemv_kernel_name_is_dimension_specialized_and_not_bf16_name")
    test_awq_backend_instance_is_distinct_from_existing_bf16_gemvs()
    print("PASS test_awq_backend_instance_is_distinct_from_existing_bf16_gemvs")
    test_placed_awq_matvec_links_awq_pythoc_object()
    print("PASS test_placed_awq_matvec_links_awq_pythoc_object")
    print("PASS test_awq_gemv_kernel_plumbing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

from kernel_builder.awq_gemv_builder import (  # noqa: E402
    awq_gemv_kernel_name,
    build_awq_gemv_air_module,
)
from kernel_builder.backend_presets import AWQ_GEMV_BACKEND  # noqa: E402


def test_awq_gemv_kernel_name_is_dimension_specialized_and_not_bf16_name():
    name = awq_gemv_kernel_name(k=2048, m=2048, group_size=128)
    assert name == "awq_gemv_k2048_m2048_g128_scalar"
    assert name not in {"rms_gemv_rope", "o_gemv_ffn", "lm_head_gemv"}
    assert awq_gemv_kernel_name(k=2048, m=2048, group_size=128, variant="vecdeq") == "awq_gemv_k2048_m2048_g128_vecdeq"


def test_awq_backend_instance_is_distinct_from_existing_bf16_gemvs():
    assert AWQ_GEMV_BACKEND["instance_name"] == "awq_gemv"
    assert AWQ_GEMV_BACKEND["instance_name"] not in {
        "rms_gemv_rope",
        "o_gemv_ffn",
        "lm_head_gemv",
    }


def test_awq_gemv_air_module_links_awq_object_and_exposes_packed_inputs():
    text = str(build_awq_gemv_air_module(k=8, m=4, group_size=4))
    assert "awq_gemv_u4_bf16" in text
    assert 'link_with = "awq_gemv_k8_m4_g4_scalar.o"' in text
    assert "memref<8xbf16>" in text
    assert "memref<16xui8>" in text
    assert "memref<16xbf16>" in text
    assert "memref<4xbf16>" in text


def main() -> int:
    test_awq_gemv_kernel_name_is_dimension_specialized_and_not_bf16_name()
    print("PASS test_awq_gemv_kernel_name_is_dimension_specialized_and_not_bf16_name")
    test_awq_backend_instance_is_distinct_from_existing_bf16_gemvs()
    print("PASS test_awq_backend_instance_is_distinct_from_existing_bf16_gemvs")
    test_awq_gemv_air_module_links_awq_object_and_exposes_packed_inputs()
    print("PASS test_awq_gemv_air_module_links_awq_object_and_exposes_packed_inputs")
    print("PASS test_awq_gemv_kernel_plumbing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

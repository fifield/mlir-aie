#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for packed-AWQ GEMV external kernel source/plumbing."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXAMPLE_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_EXAMPLE_DIR))

from kernel_builder import external_kernels  # noqa: E402


def test_awq_gemv_external_kernel_source_defines_stable_c_abi():
    src = _EXAMPLE_DIR / "kernel_builder" / "awq_gemv.cc"
    text = src.read_text(encoding="utf-8")

    assert 'extern "C"' in text
    assert "void awq_gemv_u4_bf16" in text
    assert "const bfloat16 *__restrict x" in text
    assert "const uint8_t *__restrict qweight" in text
    assert "const bfloat16 *__restrict params" in text
    assert "bfloat16 *__restrict y" in text
    assert "low nibble" in text.lower()
    assert "high nibble" in text.lower()
    assert "PackedPerGroup" in text
    assert "p_row += 2" in text
    assert "k / AWQ_GEMV_GROUP_SIZE" not in text
    assert "k / 2" not in text


def test_awq_gemv_external_kernel_vectorizes_real_group_dequant_mac_path():
    src = _EXAMPLE_DIR / "kernel_builder" / "awq_gemv.cc"
    text = src.read_text(encoding="utf-8")

    assert "#define AWQ_GEMV_VECTOR_LENGTH 32" in text
    assert "VecLen = AWQ_GEMV_VECTOR_LENGTH" in text
    assert "dequant_chunk" in text
    assert "aie::zeros<bfloat16, VecLen>()" in text
    assert "w_vec.set" in text
    assert "aie::load_v<VecLen>(x_group + chunk_start)" in text
    assert "acc = aie::mac(acc, w_vec, x_vec)" in text
    assert "AWQ_GEMV_GROUP_SIZE % VecLen" in text


def test_compile_awq_gemv_uses_distinct_object_and_dimension_defines(monkeypatch):
    calls = []

    def fake_compile(src_path, output_name, extra_flags=None, force=False):
        calls.append((Path(src_path), output_name, list(extra_flags or []), force))

    monkeypatch.setattr(external_kernels, "_compile_kernel", fake_compile)

    external_kernels.compile_awq_gemv(k=16, m=4, group_size=4, force=True)

    assert len(calls) == 1
    src_path, output_name, flags, force = calls[0]
    assert src_path.name == "awq_gemv.cc"
    assert output_name == "awq_gemv_k16_m4_g4_scalar.o"
    assert force is True
    assert "-DAWQ_GEMV_K=16" in flags
    assert "-DAWQ_GEMV_M=4" in flags
    assert "-DAWQ_GEMV_GROUP_SIZE=4" in flags
    assert "-DAWQ_GEMV_VECTORIZE_INLINE_DEQUANT=0" in flags
    assert "-DAWQ_GEMV_VECTOR_LENGTH=32" in flags


def test_compile_awq_gemv_vecdeq_uses_distinct_object_and_define(monkeypatch):
    calls = []

    def fake_compile(src_path, output_name, extra_flags=None, force=False):
        calls.append((Path(src_path), output_name, list(extra_flags or []), force))

    monkeypatch.setattr(external_kernels, "_compile_kernel", fake_compile)

    external_kernels.compile_awq_gemv(k=2048, m=32, group_size=128, variant="vecdeq")

    assert len(calls) == 1
    _, output_name, flags, _ = calls[0]
    assert output_name == "awq_gemv_k2048_m32_g128_vecdeq.o"
    assert "-DAWQ_GEMV_VECTORIZE_INLINE_DEQUANT=1" in flags
    assert "-DAWQ_GEMV_VECTOR_LENGTH=32" in flags


def test_compile_awq_gemv_rejects_unsupported_dimensions():
    try:
        external_kernels.compile_awq_gemv(k=15, m=4, group_size=4)
    except ValueError as exc:
        assert "K must be even" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected odd-K AWQ GEMV compile to fail")

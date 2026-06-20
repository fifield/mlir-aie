# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Thin entry point for the ``c2_attn`` collapsed decode device (STEP B).

``c2_attn`` = the production ``c2_merged`` o_gemv_ffn device + GQA decode
attention folded in as WAVE 0 on the row-3 (add) herd, in ONE ``aie.device`` /
ONE ``aiex.configure`` / 1 LoadPDI.  All of the actual emission lives in
``builders/o_gemv_ffn.py::_emit_call2_c2`` behind the ``attn_wave0`` flag so the
default c2_merged path is byte-for-byte untouched; this module just exposes a
seq-len-parameterized builder for the focused harness / cache.

See ``ATTN_DECODE_GQA_SCOPE.md`` "c2_attn build" for the full design.
"""

from __future__ import annotations

import os


def build_c2_attn_module(seq_len: int = 64, n_groups: int = 8, *,
                         verbose: bool = False) -> str:
    """Build the c2_attn module (seq_len baked, BF16).

    The wave-0 attention shares the add herd's DMA ring, currently wired for a
    single KV chunk -> seq_len<=64.  Multi-chunk (seq_len up to 256) is a
    follow-on (count-based cyclic K/V BD).  Per-decode-position devices keep the
    softmax mask exact.
    """
    if seq_len > 64:
        raise NotImplementedError(
            f"c2_attn wave-0 wiring is single-KV-chunk (seq_len<=64); got "
            f"{seq_len}. Multi-chunk is a follow-on.")
    if n_groups != 8:
        raise NotImplementedError(f"c2_attn fixed to 8 GQA groups; got {n_groups}")
    if verbose:
        print(f"  [c2_attn] building collapsed device seq_len={seq_len}")
    # The builder reads the seq_len from PYTHOC_C2_ATTN_SEQ_LEN.
    os.environ["PYTHOC_C2_ATTN_SEQ_LEN"] = str(seq_len)
    from .o_gemv_ffn import build_o_gemv_ffn_module
    # UNIQUE XRT kernel id per decode position so multiple per-position ELFs can
    # coexist in one process (the decode loop loads one per growing seq_len).
    return build_o_gemv_ffn_module(pack_mode="c2_attn",
                                   dispatcher_sym=c2_attn_kernel_id(seq_len))


def build_c2_attn_resident_module(n_groups: int = 8, *,
                                  verbose: bool = False) -> str:
    """Build the RESIDENT c2_attn module: ONE fixed-structure PDI reused for
    every decode position.  The trailing-chunk mask is a RUNTIME value (valid
    length L, DMA'd from the host per token) derived on-device, so a single
    ELF/PDI serves all positions -> sidesteps the two-full-fabric-PDI wedge that
    blocked the prior per-position c2_attn.

    The KV chunk ceiling is 4 (seq<=256) by default.  Setting
    ``PYTHOC_C2_ATTN_MEMKV=1`` lifts it to ``PYTHOC_C2_ATTN_MAX_CHUNKS`` (e.g.
    8 -> seq<=512) by feeding the full per-group KV in ONE shim BD/group and
    letting the add-tile fill ring backpressure it (constant shim BD usage, so
    context length no longer hits the ~16-BD shim cap).

    Behind the same ``attn_wave0`` flag (default c2_merged untouched).  Enabled
    via ``PYTHOC_C2_ATTN_RESIDENT=1``.
    """
    if n_groups != 8:
        raise NotImplementedError(f"c2_attn fixed to 8 GQA groups; got {n_groups}")
    if verbose:
        print("  [c2_attn] building RESIDENT device (runtime L)")
    os.environ["PYTHOC_C2_ATTN_SEQ_LEN"] = "256"
    os.environ["PYTHOC_C2_ATTN_RESIDENT"] = "1"
    try:
        from .o_gemv_ffn import build_o_gemv_ffn_module
        return build_o_gemv_ffn_module(
            pack_mode="c2_attn",
            dispatcher_sym=c2_attn_resident_kernel_id())
    finally:
        os.environ.pop("PYTHOC_C2_ATTN_RESIDENT", None)


def c2_attn_resident_kernel_id() -> str:
    """XRT kernel id for the single resident c2_attn device (one for all
    positions)."""
    return "o_gemv_ffn_c2attn_resident"


def c2_attn_kernel_id(seq_len: int) -> str:
    """XRT kernel id (= dispatcher runtime_sequence sym) for a c2_attn device
    at a given baked seq_len.  Used by both the builder and the host driver so
    ``main:<id>`` is unique per decode position."""
    return f"o_gemv_ffn_c2attn_s{seq_len}"

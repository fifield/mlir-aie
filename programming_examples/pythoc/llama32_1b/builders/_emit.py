# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared helpers for placed-IRON builders under ``builders/``.

These helpers were factored out of ``builders/lm_head_gemv.py`` when
adding ``builders/rms_gemv_rope.py`` -- both kernels need bf16 memrefs
with explicit memory-space attributes and the multi-launch builders
share an identical 13-arg host signature.

Nothing here is hot-path; the helpers exist purely so future Phase 4
builders (``o_gemv_ffn``, ``rms_gemms_rope``, ``o_ffn``) can pick them
up without re-implementing the boilerplate. ``lm_head_gemv.py`` was
not retrofitted to use these helpers -- that file's ``_bf16_memref``
is left in place to keep the Phase 4.1 builder bit-identical to the
known-good steady-state.
"""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16


def bf16_np(*shape):
    """Return a ``np.ndarray`` type spec with bf16 dtype + given shape.

    Used at module-construction time (outside an ``mlir_mod_ctx()``)
    to describe the host arg types of an ``aiex.runtime_sequence``.
    """
    return np.ndarray[shape, np.dtype[bfloat16]]


def bf16_memref(*shape, memory_space=None):
    """Build an ``MemRefType<...xbf16, memory_space>``.

    ``memory_space``:
        ``None`` -- L3/host buffer (no attr printed).
        ``1`` -- L2/mem tile buffer.
        ``2`` -- L1/compute tile buffer.

    Must be called inside an ``mlir_mod_ctx()`` because the underlying
    ``mlir.Type`` registers in the active context.
    """
    from aie.extras import types as T
    from aie.ir import MemRefType, IntegerAttr, IntegerType

    ms = None
    if memory_space is not None:
        ms = IntegerAttr.get(IntegerType.get_signless(32), memory_space)
    return MemRefType.get(list(shape), T.bf16(), None, ms)


# ---------------------------------------------------------------------------
# 13-arg host signature shared by all multi-launch segments of the
# rms_gemv_rope dispatcher (and, by construction, by every other
# multi-launch builder that fires under the same dispatcher).
#
# The arg layout is dictated by ``llama32_1b_decode.py``'s call into
# ``run_rms_gemv_rope`` -- see the cached IR's ``aiex.runtime_sequence
# @rms_gemv_rope(...)`` block for the canonical ordering.
# ---------------------------------------------------------------------------
def o_gemv_ffn_host_arg_types(emb_dim: int = 2048, hidden_dim: int = 8192):
    """Return the 15 host arg ``np.ndarray`` type specs for o_gemv_ffn.

    Layout (matches the cached dispatcher device's
    ``aiex.runtime_sequence @o_gemv_ffn``)::

        arg0  : memref<emb_dim x emb_dim x bf16>      wo (O proj weight)
        arg1  : memref<emb_dim x bf16>                attn_out
        arg2  : memref<emb_dim x bf16>                proj (intermediate)
        arg3  : memref<emb_dim x bf16>                x_residual
        arg4  : memref<emb_dim x bf16>                res1 (intermediate)
        arg5  : memref<emb_dim x bf16>                ffn_norm_w
        arg6  : memref<emb_dim x bf16>                normed2 (broadcast input)
        arg7  : memref<hidden_dim x emb_dim x bf16>   wgate
        arg8  : memref<hidden_dim x bf16>             gate (intermediate)
        arg9  : memref<hidden_dim x emb_dim x bf16>   wup
        arg10 : memref<hidden_dim x bf16>             up (intermediate)
        arg11 : memref<hidden_dim x bf16>             swiglu (intermediate)
        arg12 : memref<emb_dim x hidden_dim x bf16>   wdown
        arg13 : memref<emb_dim x bf16>                down (intermediate)
        arg14 : memref<emb_dim x bf16>                output
    """
    return [
        bf16_np(emb_dim, emb_dim),
        bf16_np(emb_dim),
        bf16_np(emb_dim),
        bf16_np(emb_dim),
        bf16_np(emb_dim),
        bf16_np(emb_dim),
        bf16_np(emb_dim),
        bf16_np(hidden_dim, emb_dim),
        bf16_np(hidden_dim),
        bf16_np(hidden_dim, emb_dim),
        bf16_np(hidden_dim),
        bf16_np(hidden_dim),
        bf16_np(emb_dim, hidden_dim),
        bf16_np(emb_dim),
        bf16_np(emb_dim),
    ]


def rms_gemv_rope_host_arg_types(emb_dim: int = 2048, kv_dim: int = 512):
    """Return the 13 host arg ``np.ndarray`` type specs in order.

    Layout (matches the cached dispatcher device's ``aiex.runtime_sequence
    @rms_gemv_rope``)::

        arg0  : memref<emb_dim x bf16>          rmsnorm weight (rope-K shared?)
        arg1  : memref<emb_dim x bf16>          rmsnorm in (x)
        arg2  : memref<emb_dim x bf16>          rmsnorm out (broadcast input)
        arg3  : memref<emb_dim x emb_dim x bf16>  Q weight (2048x2048)
        arg4  : memref<emb_dim x bf16>          Q output
        arg5  : memref<kv_dim x emb_dim x bf16>   K weight (512x2048)
        arg6  : memref<kv_dim x bf16>             K output
        arg7  : memref<kv_dim x emb_dim x bf16>   V weight (512x2048)
        arg8  : memref<kv_dim x bf16>             V output
        arg9  : memref<emb_dim x bf16>            RoPE-Q in (= Q out)
        arg10 : memref<kv_dim x bf16>             RoPE-K in (= K out)
        arg11 : memref<emb_dim x bf16>            RoPE-Q out
        arg12 : memref<kv_dim x bf16>             RoPE-K out
    """
    return [
        bf16_np(emb_dim),
        bf16_np(emb_dim),
        bf16_np(emb_dim),
        bf16_np(emb_dim, emb_dim),
        bf16_np(emb_dim),
        bf16_np(kv_dim, emb_dim),
        bf16_np(kv_dim),
        bf16_np(kv_dim, emb_dim),
        bf16_np(kv_dim),
        bf16_np(emb_dim),
        bf16_np(kv_dim),
        bf16_np(emb_dim),
        bf16_np(kv_dim),
    ]

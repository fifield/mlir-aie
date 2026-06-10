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
def o_gemv_ffn_host_arg_types(emb_dim: int = 2048, hidden_dim: int = 8192,
                              q_out: int | None = None):
    """Return the 15 host arg ``np.ndarray`` type specs for o_gemv_ffn.

    ``q_out`` is the O-projection contraction dim (= n_heads*head_dim, the
    attention output width). For Llama-3.2-1B it coincides with ``emb_dim``
    (2048); for Qwen3-0.6B it is 2048 while ``emb_dim`` is 1024. When ``None``
    it defaults to ``emb_dim`` (the llama coincidence), keeping the captured
    real shapes: Wo[emb,q_out], attn_out[q_out], Wg/Wu[hidden,emb], Wd[emb,hidden].

    Layout (matches the cached dispatcher device's
    ``aiex.runtime_sequence @o_gemv_ffn``)::

        arg0  : memref<emb_dim x q_out x bf16>        wo (O proj weight)
        arg1  : memref<q_out x bf16>                  attn_out
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
    if q_out is None:
        q_out = emb_dim
    return [
        bf16_np(emb_dim, q_out),
        bf16_np(q_out),
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


def rms_gemv_rope_awq_host_arg_types(emb_dim: int = 2048, kv_dim: int = 512,
                                     group_size: int = 128):
    """Return the 13 host arg type specs for ``rms_gemv_rope_awq``.

    Same layout as ``rms_gemv_rope_host_arg_types`` except args 3/5/7 (the
    Q/K/V weight matrices) are packed-AWQ ``uint8[M, K/2 + 4*groups]`` rows
    instead of ``bf16[M, K]`` rows.  RMSNorm + RoPE inputs/outputs stay bf16.

    Combined-row layout per output row: ``[qweight bytes (K/2)] [params
    bytes (4 * K/group_size)]``  --  matches ``awq_combined_weight()`` in
    ``llama32_1b_awq_runtime.py``.
    """
    row_bytes = emb_dim // 2 + 4 * (emb_dim // group_size)  # 1088 for K=2048
    u8 = np.uint8
    return [
        bf16_np(emb_dim),                                       # arg0
        bf16_np(emb_dim),                                       # arg1
        bf16_np(emb_dim),                                       # arg2
        np.ndarray[(emb_dim, row_bytes), np.dtype[u8]],         # arg3 wq AWQ
        bf16_np(emb_dim),                                       # arg4
        np.ndarray[(kv_dim, row_bytes), np.dtype[u8]],          # arg5 wk AWQ
        bf16_np(kv_dim),                                        # arg6
        np.ndarray[(kv_dim, row_bytes), np.dtype[u8]],          # arg7 wv AWQ
        bf16_np(kv_dim),                                        # arg8
        bf16_np(emb_dim),                                       # arg9
        bf16_np(kv_dim),                                        # arg10
        bf16_np(emb_dim),                                       # arg11
        bf16_np(kv_dim),                                        # arg12
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


def attach_loop_annotation_to_all_scf_for(module):
    """Walk ``module`` and attach ``loop_annotation = #llvm.loop_annotation<mustProgress = true>``
    to every ``scf.for`` op.

    The cached AIR-emitted MLIR puts this annotation on every ``scf.for``
    (via aircc's lowering pipeline).  Without it, aiecc's downstream
    lowering produces a broken ELF that emits garbage tokens.  Our
    placed-IRON builders use ``aie.helpers.dialects.scf._for`` (re-exported
    as ``range_``) which doesn't attach the annotation, so we walk the
    module ourselves before serializing to text.

    Must be called inside an active ``mlir_mod_ctx()`` (the
    ``Attribute.parse`` call needs the same context as ``module``).
    """
    from aie.ir import Attribute

    annot = Attribute.parse("#llvm.loop_annotation<mustProgress = true>")

    def walk(op):
        if op.operation.name == "scf.for":
            op.operation.attributes["loop_annotation"] = annot
        for region in op.regions:
            for block in region:
                for sub in block:
                    walk(sub)

    for op in module.body:
        walk(op)


def matvec_herd_descriptors(out_rows, k_dim, n_cols, m_tile, rows_per_outer_cap=1024):
    """Output + weight DMA descriptors for an `n_cols`-column GEMV herd that
    computes ``[out_rows, k_dim] @ x[k_dim] -> y[out_rows]``, splitting `out_rows`
    across `n_cols` columns x `m_tile`-row bands and looping `n_outer` times over
    `rows_per_outer` rows (capped at `rows_per_outer_cap` to bound the per-outer
    L1 footprint). `k_dim` is the contraction (= the host weight's inner dim and
    the broadcast-X length).

    Returns a dict of the descriptors the matvec-seg runtime_sequence needs.
    Verified to reproduce the llama hand-tuned constants exactly:
      rms_gemv_rope K/V (out=512,k=2048,m=8) -> y_dims=[(8,64),(8,1)], w_dims=[(8,131072),(32,512),(512,1)]
      rms_gemv_rope Q   (out=2048,k=2048,m=8) -> y_dims=[(16,64),(8,1)], w_len=262144, weight_outer_stride=1024*2048
    Shapes must satisfy: out_rows % n_outer == 0; rows_per_outer % (n_cols*m_tile) == 0;
    (m_tile*k_dim) % 512 == 0 (512 = the fixed broadcast chunk).
    """
    n_outer = max(1, -(-out_rows // rows_per_outer_cap))  # ceil
    if out_rows % n_outer:
        raise ValueError(f"out_rows={out_rows} not divisible into n_outer={n_outer}")
    rows_per_outer = out_rows // n_outer
    band = n_cols * m_tile
    if rows_per_outer % band:
        raise ValueError(f"rows_per_outer={rows_per_outer} not divisible by n_cols*m_tile={band}")
    mtile_k = m_tile * k_dim
    if mtile_k % 512:
        raise ValueError(f"m_tile*k_dim={mtile_k} not divisible by the 512 broadcast chunk")
    size_outer = rows_per_outer // band
    return {
        "n_outer": n_outer,
        "rows_per_outer": rows_per_outer,
        "size_outer": size_outer,
        "y_len": rows_per_outer // n_cols,
        "y_dims": [(size_outer, band), (m_tile, 1)],
        "x_repeat_count": 2 * size_outer - 1,
        "w_dims": [(size_outer, n_cols * mtile_k), (mtile_k // 512, 512), (512, 1)],
        "w_len": size_outer * mtile_k,
        "weight_col_stride": mtile_k,
        "weight_outer_stride": (rows_per_outer * k_dim) if n_outer > 1 else 0,
        "output_col_stride": m_tile,
        "output_outer_stride": rows_per_outer if n_outer > 1 else 0,
    }

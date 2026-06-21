# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Experimental runtime helpers for direct packed-AWQ NPU primitives.

These helpers are intentionally opt-in and are not used by the normal Llama
AWQ path yet. The existing --quant awq flow still CPU-dequantizes weights and
runs the BF16 kernels until fused AWQ integration is validated projection by
projection.
"""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

from kernel_builder.aie_ir_gen import build_awq_gemv_ir
from kernel_builder.backend_presets import AWQ_GEMV_BACKEND, OGF_AWQ_BACKEND
from kernel_builder.external_kernels import awq_gemv_kernel_name
from llama32_1b_weights import AwqLinear


def _validate_awq_gemv_inputs(x_bf16, awq: AwqLinear) -> np.ndarray:
    """Validate direct AWQ GEMV inputs and return a contiguous BF16 vector."""
    x = np.ascontiguousarray(np.asarray(x_bf16, dtype=bfloat16).reshape(-1))
    if x.shape[0] != awq.k:
        raise ValueError(f"x length {x.shape[0]} does not match AWQ K={awq.k}")

    expected_q_shape = (awq.m, awq.k // 2)
    if awq.qweight.shape != expected_q_shape:
        raise ValueError(
            f"AWQ qweight shape {awq.qweight.shape} does not match expected {expected_q_shape}"
        )

    groups = awq.k // awq.group_size
    expected_p_shape = (awq.m, 2 * groups)
    if awq.params.shape != expected_p_shape:
        raise ValueError(
            f"AWQ params shape {awq.params.shape} does not match expected {expected_p_shape}"
        )
    return x


def _ensure_awq_gemv_compiled(cache, name: str, k: int, m: int, group_size: int, *, variant: str = "vecdeq") -> None:
    """Compile an AWQ GEMV kernel unless this cache already has it."""
    if name in getattr(cache, "artifacts", {}):
        return
    cache.compile_and_cache(
        name,
        build_awq_gemv_ir(k, m, group_size, variant=variant),
        AWQ_GEMV_BACKEND["instance_name"],
    )


def awq_gemv_npu(cache, x_bf16, awq: AwqLinear, *, variant: str = "vecdeq") -> np.ndarray:
    """Run one direct packed uint4 AWQ GEMV on NPU.

    Args:
        cache: KernelCache-like object used to compile/load/run the kernel.
        x_bf16: bf16-compatible vector of length awq.k.
        awq: packed AWQ linear with qweight [M, K/2] and params [M, 2*groups].

    Returns:
        bf16 vector of length awq.m.

    This is a correctness-first primitive wrapper. It consumes AwqLinear.qweight
    and AwqLinear.params directly and never materializes a full BF16 weight
    matrix. It is deliberately not wired into the default decode path yet.
    """
    x = _validate_awq_gemv_inputs(x_bf16, awq)

    name = awq_gemv_kernel_name(awq.k, awq.m, awq.group_size, variant=variant)
    _ensure_awq_gemv_compiled(cache, name, awq.k, awq.m, awq.group_size, variant=variant)

    y = np.zeros((awq.m,), dtype=bfloat16)
    results = cache.load_and_run(
        name,
        AWQ_GEMV_BACKEND,
        x,
        np.asarray(awq.qweight, dtype=np.uint8).reshape(-1),
        np.asarray(awq.params, dtype=bfloat16).reshape(-1),
        y,
        output_indices=[3],
    )
    return np.asarray(results[3], dtype=bfloat16).reshape(awq.m)


def awq_gemv_npu_tiled(cache, x_bf16, awq: AwqLinear, *, tile_m: int, variant: str = "vecdeq") -> np.ndarray:
    """Run a full packed-AWQ GEMV by chunking output rows into NPU tiles.

    This covers full model-sized projections with the current correctness-first
    primitive, whose single invocation stages one row block of qweight/params
    into L1. The full BF16 weight matrix is never materialized; each tile passes
    only a slice of ``awq.qweight`` and ``awq.params`` to ``awq_gemv_npu``.
    """
    if tile_m <= 0:
        raise ValueError(f"tile_m must be positive, got {tile_m}")
    x = _validate_awq_gemv_inputs(x_bf16, awq)
    out = np.empty((awq.m,), dtype=bfloat16)
    for row_start in range(0, awq.m, tile_m):
        row_end = min(awq.m, row_start + tile_m)
        tile = AwqLinear(
            qweight=np.ascontiguousarray(awq.qweight[row_start:row_end], dtype=np.uint8),
            params=np.ascontiguousarray(awq.params[row_start:row_end], dtype=bfloat16),
            k=awq.k,
            m=row_end - row_start,
            group_size=awq.group_size,
        )
        out[row_start:row_end] = awq_gemv_npu(cache, x, tile, variant=variant)
    return out


def awq_combined_weight(awq: AwqLinear) -> np.ndarray:
    """Pack (qweight, params) into the combined uint8 row layout the AIR
    matvec consumes. Each row holds [qweight_bytes (K/2)] [params bytes].
    Cached on the AwqLinear instance via attribute ``_combined``.
    """
    cached = getattr(awq, "_combined", None)
    if cached is not None and cached.shape == (awq.m, awq.k // 2 + 4 * (awq.k // awq.group_size)):
        return cached
    groups = awq.k // awq.group_size
    row_bytes = awq.k // 2 + 4 * groups
    q = np.ascontiguousarray(awq.qweight, dtype=np.uint8).reshape(awq.m, awq.k // 2)
    p = np.ascontiguousarray(awq.params, dtype=bfloat16).reshape(awq.m, 2 * groups)
    combined = np.empty((awq.m, row_bytes), dtype=np.uint8)
    combined[:, : awq.k // 2] = q
    combined[:, awq.k // 2 :] = p.view(np.uint8).reshape(awq.m, 4 * groups)
    awq._combined = combined
    return combined


def _ensure_o_gemv_ffn_awq_compiled(cache, emb_dim: int, hidden_dim: int, group_size: int) -> None:
    """Compile the fused packed-AWQ O+FFN decode kernel unless cached.

    The PythoC AWQ kernels (awq_mv_pythoc.o, awq_mv_k8192_pythoc.o) are
    registered in ``kernel_builder.external_kernels._PYTHOC_KERNELS`` and
    compiled lazily by ``_stage_required_objs`` during cache.compile_and_cache.
    GROUP_SIZE=128 is baked into the kernel sources; runtime-variable
    group_size is no longer supported (matches the AIR-tree convention
    where group_size was a -D macro at compile time).
    """
    del group_size  # baked into kernels/awq_mv.py (GROUP_SIZE: i32 = 128)
    name = "o_gemv_ffn_awq"
    if name in getattr(cache, "artifacts", {}):
        return
    from kernel_builder.aie_ir_gen import (build_o_gemv_ffn_awq_ir,
                                           o_gemv_ffn_awq_pack_mode)

    cache.compile_and_cache(
        name,
        build_o_gemv_ffn_awq_ir(emb_dim, hidden_dim, group_size=128),
        OGF_AWQ_BACKEND["instance_name"],
        config={"pack_mode": o_gemv_ffn_awq_pack_mode()},
    )
    # Persist the lazily-compiled AWQ ELF so the next process reuses it instead
    # of recompiling (~7s). The pack-mode config makes a c2_merged<->c2_attn
    # toggle invalidate the shared slot. _save_manifest merges, so the BF16
    # entries already in the manifest are preserved.
    cache._save_manifest()


def o_gemv_ffn_awq_npu(
    cache,
    attn_out,
    x_bf16,
    ffn_norm_w,
    awq_layer_weights,
    *,
    emb_dim: int,
    hidden_dim: int,
    layer_idx=None,
) -> np.ndarray:
    """Run fused packed-AWQ O projection + FFN decode in one XRT call.

    Each AWQ linear is sent as a single combined uint8 buffer (qweight rows
    followed by params bytes per row); this fits within the device's shim
    DMA channel budget after the fused module is lowered.
    """
    awqs = [
        awq_layer_weights.wo,
        awq_layer_weights.w_gate,
        awq_layer_weights.w_up,
        awq_layer_weights.w_down,
    ]
    group_sizes = {int(a.group_size) for a in awqs}
    if len(group_sizes) != 1:
        raise ValueError(f"Fused AWQ O+FFN requires one group size, got {sorted(group_sizes)}")
    group_size = group_sizes.pop()

    _validate_awq_gemv_inputs(attn_out, awq_layer_weights.wo)
    norm_probe = np.zeros((emb_dim,), dtype=bfloat16)
    _validate_awq_gemv_inputs(norm_probe, awq_layer_weights.w_gate)
    _validate_awq_gemv_inputs(norm_probe, awq_layer_weights.w_up)
    swiglu_probe = np.zeros((hidden_dim,), dtype=bfloat16)
    _validate_awq_gemv_inputs(swiglu_probe, awq_layer_weights.w_down)

    _ensure_o_gemv_ffn_awq_compiled(cache, emb_dim, hidden_dim, group_size)

    wo_w = awq_combined_weight(awq_layer_weights.wo)
    wgate_w = awq_combined_weight(awq_layer_weights.w_gate)
    wup_w = awq_combined_weight(awq_layer_weights.w_up)
    wdown_w = awq_combined_weight(awq_layer_weights.w_down)

    proj_buf = np.zeros(emb_dim, dtype=bfloat16)
    x_residual = np.ascontiguousarray(np.asarray(x_bf16, dtype=bfloat16).reshape(emb_dim))
    res1_buf = np.zeros(emb_dim, dtype=bfloat16)
    ffn_norm = np.ascontiguousarray(np.asarray(ffn_norm_w, dtype=bfloat16).reshape(emb_dim))
    normed2_buf = np.zeros(emb_dim, dtype=bfloat16)
    gate_buf = np.zeros(hidden_dim, dtype=bfloat16)
    up_buf = np.zeros(hidden_dim, dtype=bfloat16)
    swiglu_buf = np.zeros(hidden_dim, dtype=bfloat16)
    down_buf = np.zeros(emb_dim, dtype=bfloat16)
    output_buf = np.zeros(emb_dim, dtype=bfloat16)

    bo_key = f"o_gemv_ffn_awq_L{layer_idx}" if layer_idx is not None else None
    results = cache.load_and_run(
        "o_gemv_ffn_awq",
        OGF_AWQ_BACKEND,
        wo_w,                                                                # 0 static
        np.ascontiguousarray(np.asarray(attn_out, dtype=bfloat16).reshape(emb_dim)),  # 1
        proj_buf,                                                            # 2 intermediate
        x_residual,                                                          # 3
        res1_buf,                                                            # 4 intermediate
        ffn_norm,                                                            # 5 static
        normed2_buf,                                                         # 6 intermediate
        wgate_w,                                                             # 7 static
        gate_buf,                                                            # 8 intermediate
        wup_w,                                                               # 9 static
        up_buf,                                                              # 10 intermediate
        swiglu_buf,                                                          # 11 intermediate
        wdown_w,                                                             # 12 static
        down_buf,                                                            # 13 intermediate
        output_buf,                                                          # 14 output
        output_indices=[14],
        static_input_indices={0, 5, 7, 9, 12},
        intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14},
        bo_key=bo_key,
    )
    return np.asarray(results[14], dtype=bfloat16).reshape(emb_dim)


def o_gemv_ffn_awq_c2_attn_npu(
    cache,
    layer_weights,
    awq_layer_weights,
    x_bf16,
    q_roped,
    k_cache_layer,
    v_cache_layer,
    current_pos,
    *,
    emb_dim: int,
    hidden_dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    layer_idx=None,
) -> np.ndarray:
    """AWQ counterpart of llama32_1b_decode._run_c2_attn.

    Runs call 2 under the RESIDENT AWQ ``c2_attn`` collapsed device: GQA decode
    attention as WAVE 0 on the row-3 add herd (weight-free BFP576 BF16 kernels),
    folded into the uint4 O+add1+gate/up+swiglu+down+add2 device.  ONE resident
    PDI for all positions/layers; the trailing-chunk softmax mask is a RUNTIME
    valid length L = current_pos+1 folded into q's free padding (untiled offset
    256).  q + this layer's K/V cache pack into the 18-arg extended AWQ ABI:
    base AWQ 15 args (uint4 weights, arg1 widened to the per-group attn-out
    scratch) + q_all/k_all/v_all.

    Reuses the weight-free host KV tiling from llama32_1b_decode (incremental
    per-token tile, no O(seq) re-pack) so the host packer is single-sourced with
    the BF16 c2_attn path.
    """
    from kernel_builder.aie_ir_gen import (build_o_gemv_ffn_awq_ir,
                                           o_gemv_ffn_awq_pack_mode)
    import llama32_1b_decode as _dec

    group_size = n_heads // n_kv_heads
    seq_len = current_pos + 1
    RES_MAX_CHUNKS = _dec._RES_MAX_CHUNKS
    RES_PADDED = _dec._RES_PADDED
    assert seq_len <= RES_PADDED, (
        f"AWQ c2_attn supports seq_len<={RES_PADDED} "
        f"(MAX_CHUNKS={RES_MAX_CHUNKS}, MEMKV={_dec._C2_ATTN_MEMKV}); got "
        f"{seq_len} (current_pos={current_pos})")
    TILE_ROWS = _dec._RES_TILE_ROWS
    tile_size = TILE_ROWS * head_dim
    kv_size = RES_MAX_CHUNKS * tile_size

    # ONE resident AWQ PDI for ALL positions/layers, built into the
    # o_gemv_ffn_awq cache slot under its instance name (the pack-mode env makes
    # build_o_gemv_ffn_awq_ir emit the c2_attn device).
    name = "o_gemv_ffn_awq"
    if name not in getattr(cache, "artifacts", {}):
        cache.compile_and_cache(
            name,
            build_o_gemv_ffn_awq_ir(emb_dim, hidden_dim, group_size=128),
            OGF_AWQ_BACKEND["instance_name"],
            config={"pack_mode": o_gemv_ffn_awq_pack_mode()},
        )
        # Persist so the next process reuses the ELF instead of recompiling the
        # resident c2_attn device (~7s) on the first decode token. The pack-mode
        # config invalidates the shared o_gemv_ffn_awq slot on a mode switch.
        cache._save_manifest()

    qh = np.asarray(q_roped, dtype=bfloat16).reshape(n_heads, head_dim)
    q_all = np.zeros(n_kv_heads * tile_size, dtype=bfloat16)

    # --- INCREMENTAL tiled K/V (weight-free; shared verbatim with the BF16
    # c2_attn host packer so the on-host KV tiling cannot drift). ---
    state_key = layer_idx if layer_idx is not None else id(k_cache_layer)
    st = _dec._C2_ATTN_KV_STATE.get(state_key)
    if st is None:
        k_all = np.zeros(n_kv_heads * kv_size, dtype=bfloat16)
        v_all = np.zeros(n_kv_heads * kv_size, dtype=bfloat16)
        seed_pos = current_pos
        for g in range(n_kv_heads):
            for c in range(RES_MAX_CHUNKS):
                lo = c * TILE_ROWS
                hi = min(lo + TILE_ROWS, seed_pos)
                if hi <= lo:
                    break
                k_pad = np.zeros((TILE_ROWS, head_dim), dtype=bfloat16)
                v_pad = np.zeros((TILE_ROWS, head_dim), dtype=bfloat16)
                k_pad[:hi - lo] = k_cache_layer[g, lo:hi, :].astype(bfloat16)
                v_pad[:hi - lo] = v_cache_layer[g, lo:hi, :].astype(bfloat16)
                base = g * kv_size + c * tile_size
                k_all[base:base + tile_size] = _dec._c2_attn_tile_8x8(k_pad)
                v_all[base:base + tile_size] = _dec._c2_attn_tile_8x8(v_pad)
        st = {"k_all": k_all, "v_all": v_all, "seeded_pos": seed_pos}
        _dec._C2_ATTN_KV_STATE[state_key] = st
    k_all = st["k_all"]
    v_all = st["v_all"]

    for s in range(st["seeded_pos"], current_pos + 1):
        c = s // TILE_ROWS
        r = s % TILE_ROWS
        off = _dec._C2_ATTN_ROW_OFF + r * 8
        for g in range(n_kv_heads):
            base = g * kv_size + c * tile_size
            k_all[base + off] = k_cache_layer[g, s, :].astype(bfloat16)
            v_all[base + off] = v_cache_layer[g, s, :].astype(bfloat16)
    st["seeded_pos"] = current_pos + 1

    for g in range(n_kv_heads):
        q_pad = np.zeros((TILE_ROWS, head_dim), dtype=bfloat16)
        q_pad[:group_size] = qh[g * group_size:(g + 1) * group_size]
        q_all[g * tile_size:(g + 1) * tile_size] = q_pad.reshape(-1)
        # Fold runtime valid length L = seq_len into q's free padding (untiled
        # offset 256 = tiled (row4,col0); rows 0..3 hold the real heads).
        q_all[g * tile_size + 256] = bfloat16(float(seq_len))

    wo_w = awq_combined_weight(awq_layer_weights.wo)
    wgate_w = awq_combined_weight(awq_layer_weights.w_gate)
    wup_w = awq_combined_weight(awq_layer_weights.w_up)
    wdown_w = awq_combined_weight(awq_layer_weights.w_down)

    attn_scratch = np.zeros(n_kv_heads * tile_size, dtype=bfloat16)  # arg1 wide
    proj_buf = np.zeros(emb_dim, dtype=bfloat16)
    x_residual = np.ascontiguousarray(
        np.asarray(x_bf16, dtype=bfloat16).reshape(emb_dim))
    res1_buf = np.zeros(emb_dim, dtype=bfloat16)
    ffn_norm = np.ascontiguousarray(
        layer_weights.ffn_norm.reshape(emb_dim).astype(bfloat16))
    normed2_buf = np.zeros(emb_dim, dtype=bfloat16)
    gate_buf = np.zeros(hidden_dim, dtype=bfloat16)
    up_buf = np.zeros(hidden_dim, dtype=bfloat16)
    swiglu_buf = np.zeros(hidden_dim, dtype=bfloat16)
    down_buf = np.zeros(emb_dim, dtype=bfloat16)
    output_buf = np.zeros(emb_dim, dtype=bfloat16)

    bo_key = (f"c2attn_o_gemv_ffn_awq_L{layer_idx}"
              if layer_idx is not None else None)
    results = cache.load_and_run(
        name,
        OGF_AWQ_BACKEND,
        wo_w, attn_scratch, proj_buf, x_residual, res1_buf, ffn_norm,
        normed2_buf, wgate_w, gate_buf, wup_w, up_buf, swiglu_buf, wdown_w,
        down_buf, output_buf,
        q_all, k_all, v_all,
        output_indices=[14],
        static_input_indices={0, 5, 7, 9, 12},
        intermediate_indices={1, 2, 4, 6, 8, 10, 11, 13, 14},
        bo_key=bo_key,
    )
    return np.asarray(results[14], dtype=bfloat16).reshape(emb_dim)

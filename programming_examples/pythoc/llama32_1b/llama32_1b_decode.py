# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLAMA-3.2-1B Decode on MLIR-AIE (NPU2).

Single-token autoregressive generation with KV cache.
Runs prefill first to populate KV cache, then decodes token-by-token.

Usage:
    cd build_peano
    python3 ../llama32_1b_decode.py --compile-only
    python3 ../llama32_1b_decode.py --run-only --n-tokens 10 --profile
    python3 ../llama32_1b_decode.py --run-only --n-tokens 1 --verify
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.dirname(__file__))

from llama32_1b_weights import LlamaConfig
from kernel_builder.cache import KernelCache
from kernel_builder.backend_presets import (
    RGR_BACKEND,
    RGR_AWQ_BACKEND,
    OGF_BACKEND,
    LM_GEMV_BACKEND,
)

# ---------------------------------------------------------------------------
# Decode kernel compilation
# ---------------------------------------------------------------------------


def decode_cache_signatures():
    """Per-kernel build signatures recorded in the decode manifest.

    Currently only the device-packing mode affects the emitted IR, so the
    signature is the resolved pack mode for each packable decode kernel. A
    `make run` after toggling `PYTHOC_LLAMA_*_PACK_MODE` sees the mismatch via
    `KernelCache.load_manifest(expected_configs=...)` and rebuilds.
    """
    from kernel_builder import aie_ir_gen

    modes = aie_ir_gen.decode_pack_modes()
    return {
        "rms_gemv_rope": {"pack_mode": modes["rms_gemv_rope"]},
        "o_gemv_ffn": {"pack_mode": modes["o_gemv_ffn"]},
    }


def compile_decode_kernels(cache, config):
    """Compile the 3 merged decode kernels (mlir-aie -> aiecc -> ELF)."""
    from kernel_builder.external_kernels import compile_all_external_kernels
    from kernel_builder import aie_ir_gen

    compile_all_external_kernels(head_dim=config.head_dim)

    emb_dim = config.emb_dim
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    n_heads = config.n_heads
    kv_dim = n_kv_heads * head_dim

    sigs = decode_cache_signatures()

    print(f"\n{'='*60}")
    print(f"Compiling decode kernels (2-call merged pipeline)...")
    print(f"  pack modes: rms_gemv_rope={sigs['rms_gemv_rope']['pack_mode']}, "
          f"o_gemv_ffn={sigs['o_gemv_ffn']['pack_mode']}")
    print(f"{'='*60}\n")

    cache.compile_and_cache(
        "rms_gemv_rope",
        aie_ir_gen.build_rms_gemv_rope_ir(
            emb_dim, kv_dim, n_heads, n_kv_heads, head_dim,
            verbose=cache.verbose,
        ),
        instance_name="rms_gemv_rope",
        config=sigs["rms_gemv_rope"],
    )

    if sigs["o_gemv_ffn"]["pack_mode"] == "c2_attn":
        # c2_attn folds attention into call 2 as wave 0 and runs the RESIDENT
        # device (ONE fixed 4-chunk PDI for all positions/layers, runtime L).
        # Build it into the o_gemv_ffn cache slot under the resident XRT kernel
        # id so --run-only's manifest/pack-mode check is satisfied and
        # _run_c2_attn loads the same artifact (no lazy rebuild).
        from builders.c2_attn import (build_c2_attn_resident_module,
                                      c2_attn_resident_kernel_id)
        cache.compile_and_cache(
            "o_gemv_ffn",
            build_c2_attn_resident_module(n_kv_heads, verbose=cache.verbose),
            instance_name=c2_attn_resident_kernel_id(),
            config=sigs["o_gemv_ffn"],
        )
    else:
        cache.compile_and_cache(
            "o_gemv_ffn",
            aie_ir_gen.build_o_gemv_ffn_ir(emb_dim, hidden_dim,
                                           verbose=cache.verbose),
            instance_name="o_gemv_ffn",
            config=sigs["o_gemv_ffn"],
        )

    cache.compile_and_cache(
        "lm_head_gemv",
        aie_ir_gen.build_lm_head_gemv_ir(emb_dim, verbose=cache.verbose),
        instance_name="lm_head_gemv",
    )

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} decode kernels compiled.")


# ---------------------------------------------------------------------------
# CPU decode attention (with KV cache)
# ---------------------------------------------------------------------------


def decode_attention_cpu(
    q, k_cache, v_cache, current_pos, n_heads, n_kv_heads, head_dim
):
    """Single-query attention with KV cache.

    Args:
        q: (emb_dim,) — query vector for current token
        k_cache: (n_kv_heads, max_seq, head_dim) — cached keys [0:current_pos+1]
        v_cache: (n_kv_heads, max_seq, head_dim) — cached values [0:current_pos+1]
        current_pos: current token position (0-indexed)
        n_heads: number of Q heads (32)
        n_kv_heads: number of KV heads (8)
        head_dim: head dimension (64)

    Returns:
        attn_out: (emb_dim,) — attention output
    """
    group_size = n_heads // n_kv_heads
    scale = 1.0 / np.sqrt(head_dim)
    seq_len = current_pos + 1

    q_heads = q.astype(np.float32).reshape(n_heads, head_dim)
    k_cached = k_cache[:, :seq_len, :].astype(np.float32)  # (n_kv, seq, hd)
    v_cached = v_cache[:, :seq_len, :].astype(np.float32)

    out = np.zeros((n_heads, head_dim), dtype=np.float32)
    for h in range(n_heads):
        kv_h = h // group_size
        scores = (q_heads[h] @ k_cached[kv_h].T) * scale  # (seq,)
        probs = np.exp(scores - scores.max())
        probs = probs / probs.sum()
        out[h] = probs @ v_cached[kv_h]  # (hd,)

    return out.reshape(-1).astype(bfloat16)


# ---------------------------------------------------------------------------
# Single decode transformer block
# ---------------------------------------------------------------------------


def run_decode_block(
    x_bf16,
    layer_weights,
    cache,
    config,
    k_cache_layer,
    v_cache_layer,
    current_pos,
    rope_lut_bf16,
    awq_layer_weights=None,
    awq_tile_m_k2048=32,
    awq_tile_m_k8192=8,
):
    """Run one transformer block for a single decode token.

    Args:
        x_bf16: (emb_dim,) input token embedding
        layer_weights: LayerWeights for this layer
        cache: KernelCache
        config: LlamaConfig
        k_cache_layer: (n_kv_heads, max_seq, head_dim) — this layer's K cache
        v_cache_layer: (n_kv_heads, max_seq, head_dim) — this layer's V cache
        current_pos: current token position
        rope_lut_bf16: (max_seq, head_dim) RoPE LUT

    Returns:
        output: (emb_dim,) — block output
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    layer_idx = None  # Set by caller via layer_weights._layer_idx
    if hasattr(layer_weights, "_layer_idx"):
        layer_idx = layer_weights._layer_idx

    def _run(name, backend, *inputs, static_indices=None, **kwargs):
        # Per-layer BO key: same XRT context, separate BOs for weight isolation
        bk = (
            f"{name}_L{layer_idx}" if static_indices and layer_idx is not None else None
        )
        return cache.load_and_run(
            name,
            backend,
            *inputs,
            bo_key=bk,
            static_input_indices=static_indices,
            **kwargs,
        )

    # --- Call 1: rms_gemv_rope (6 launches, 13 args) ---
    # RMSNorm + Q/K/V GEMV + RoPE Q + RoPE K.  When awq_layer_weights is
    # provided, dispatch to the AWQ variant which reads packed-uint4
    # qweight+params for Q/K/V instead of bf16 weight matrices.
    x_in = x_bf16.flatten().astype(bfloat16)
    w_norm = layer_weights.attn_norm.reshape(emb_dim).astype(bfloat16)
    normed_buf = np.zeros(emb_dim, dtype=bfloat16)
    if awq_layer_weights is not None:
        from llama32_1b_awq_runtime import awq_combined_weight
        wq = awq_combined_weight(awq_layer_weights.wq)
        wk = awq_combined_weight(awq_layer_weights.wk)
        wv = awq_combined_weight(awq_layer_weights.wv)
        rgr_kernel = "rms_gemv_rope_awq"
        rgr_backend = RGR_AWQ_BACKEND
    else:
        wq = layer_weights._wq_t
        wk = layer_weights._wk_t
        wv = layer_weights._wv_t
        rgr_kernel = "rms_gemv_rope"
        rgr_backend = RGR_BACKEND
    q_buf = np.zeros(emb_dim, dtype=bfloat16)
    k_buf = np.zeros(kv_dim, dtype=bfloat16)
    v_buf = np.zeros(kv_dim, dtype=bfloat16)

    # RoPE LUT for current position
    rope_lut_pos = rope_lut_bf16[current_pos : current_pos + 1]  # (1, 64)
    lut_q = np.tile(rope_lut_pos, (n_heads, 1)).flatten().astype(bfloat16)
    lut_k = np.tile(rope_lut_pos, (n_kv_heads, 1)).flatten().astype(bfloat16)
    q_roped_buf = np.zeros(emb_dim, dtype=bfloat16)
    k_roped_buf = np.zeros(kv_dim, dtype=bfloat16)

    results = _run(
        rgr_kernel,
        rgr_backend,
        x_in,  # arg0
        w_norm,  # arg1
        normed_buf,  # arg2 (intermediate)
        wq,  # arg3 (static)
        q_buf,  # arg4 (intermediate)
        wk,  # arg5 (static)
        k_buf,  # arg6 (intermediate)
        wv,  # arg7 (static)
        v_buf,  # arg8 (intermediate/output)
        lut_q,  # arg9
        lut_k,  # arg10
        q_roped_buf,  # arg11 (intermediate/output)
        k_roped_buf,  # arg12 (intermediate/output)
        output_indices=[8, 11, 12],
        static_indices={3, 5, 7},
        intermediate_indices={2, 4, 6, 8, 11, 12},
    )
    v = results[8].astype(bfloat16)
    q_roped = results[11].reshape(n_heads, head_dim).astype(bfloat16)
    k_roped = results[12].reshape(n_kv_heads, head_dim).astype(bfloat16)

    # Update KV cache
    k_cache_layer[:, current_pos, :] = k_roped
    v_cache_layer[:, current_pos, :] = v.reshape(n_kv_heads, head_dim)

    # --- Attention: CPU by default; on-NPU wave 0 under o_gemv_ffn=c2_attn ---
    from kernel_builder import aie_ir_gen as _air
    _ogf_mode = _air.decode_pack_modes().get("o_gemv_ffn")
    _c2_attn = (_ogf_mode == "c2_attn") and awq_layer_weights is None
    if _c2_attn:
        # c2_attn folds attention into call 2 as wave 0; no CPU attention.
        attn_out = None
    else:
        attn_out = decode_attention_cpu(
            q_roped.flatten(),
            k_cache_layer,
            v_cache_layer,
            current_pos,
            n_heads,
            n_kv_heads,
            head_dim,
        )

    # --- Call 2: o_gemv_ffn (8 launches, 15 args) ---
    # O GEMV + Add + RMSNorm + Gate/Up GEMV + SiLU*mul + Down GEMV + Add
    if awq_layer_weights is not None:
        from llama32_1b_awq_runtime import o_gemv_ffn_awq_npu
        return o_gemv_ffn_awq_npu(
            cache,
            attn_out,
            x_bf16,
            layer_weights.ffn_norm.reshape(emb_dim).astype(bfloat16),
            awq_layer_weights,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            layer_idx=layer_idx,
        )

    if _c2_attn:
        return _run_c2_attn(
            cache, layer_weights, x_bf16, q_roped, k_cache_layer, v_cache_layer,
            current_pos, emb_dim, hidden_dim, n_heads, n_kv_heads, head_dim)

    wo = layer_weights._wo_t
    proj_buf = np.zeros(emb_dim, dtype=bfloat16)
    x_residual = x_bf16.flatten().astype(bfloat16)
    res1_buf = np.zeros(emb_dim, dtype=bfloat16)
    w_norm2 = layer_weights.ffn_norm.reshape(emb_dim).astype(bfloat16)
    normed2_buf = np.zeros(emb_dim, dtype=bfloat16)
    w_gate = layer_weights._wgate_t
    gate_buf = np.zeros(hidden_dim, dtype=bfloat16)
    w_up = layer_weights._wup_t
    up_buf = np.zeros(hidden_dim, dtype=bfloat16)
    swiglu_buf = np.zeros(hidden_dim, dtype=bfloat16)
    w_down = layer_weights._wdown_t
    down_buf = np.zeros(emb_dim, dtype=bfloat16)
    output_buf = np.zeros(emb_dim, dtype=bfloat16)

    results = _run(
        "o_gemv_ffn",
        OGF_BACKEND,
        wo,  # arg0 (static)
        attn_out,  # arg1
        proj_buf,  # arg2 (intermediate)
        x_residual,  # arg3
        res1_buf,  # arg4 (intermediate)
        w_norm2,  # arg5
        normed2_buf,  # arg6 (intermediate)
        w_gate,  # arg7 (static)
        gate_buf,  # arg8 (intermediate)
        w_up,  # arg9 (static)
        up_buf,  # arg10 (intermediate)
        swiglu_buf,  # arg11 (intermediate)
        w_down,  # arg12 (static)
        down_buf,  # arg13 (intermediate)
        output_buf,  # arg14 (intermediate/output)
        output_indices=[14],
        static_indices={0, 7, 9, 12},
        intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14},
    )
    output = results[14].astype(bfloat16)

    return output


_C2_ATTN_BUILT = set()

# Resident c2_attn: ONE fixed-structure PDI (MAX_CHUNKS=4, seq_len<=256) reused
# for every decode position; trailing-chunk mask is a runtime value L.
_RES_MAX_CHUNKS = 4
_RES_TILE_ROWS = 64
_RES_PADDED = _RES_MAX_CHUNKS * _RES_TILE_ROWS  # 256


def _c2_attn_tile_8x8(mat):
    """Tile a (64,64) matrix into 8x8 column-block-major flat layout:
    element (row,col) -> flat (col//8)*512 + row*8 + (col%8).  Matches the
    on-device q/k/v tiled buffer the matmul kernels read (mirrors
    tools/test_c2_attn.py::_tile_8x8).

    Vectorized: element (row,col)->flat (col//8)*512 + row*8 + (col%8) is a pure
    permutation. cb=col//8, ci=col%8 -> out[cb,row,ci] = mat[row, cb*8+ci] =
    mat.reshape(64,8,8)[row,cb,ci], i.e. transpose(1,0,2). Bit-identical to the
    old 64-iter Python loop (verified), ~60x faster (3.3us vs 196us/call) -- the
    loop was ~200 ms/token of host overhead (1024 calls/token)."""
    return np.ascontiguousarray(
        mat.reshape(64, 8, 8).transpose(1, 0, 2)).reshape(-1)


def _run_c2_attn(cache, layer_weights, x_bf16, q_roped, k_cache_layer,
                 v_cache_layer, current_pos, emb_dim, hidden_dim,
                 n_heads, n_kv_heads, head_dim):
    """Call 2 under the RESIDENT c2_attn collapsed device (attention as wave 0).

    Packs q_roped + this layer's K/V cache into the resident batched-attention
    BO layout (arg15/16/17), widens arg1 to the per-group attn_out scratch, and
    feeds the 18-arg c2_attn ABI.  Uses ONE PDI for ALL positions/layers
    (``o_gemv_ffn_c2attn_resident``): fixed 4-chunk structure (KV zero-padded to
    256), the trailing-chunk softmax mask is a RUNTIME valid-length L =
    current_pos+1 folded into q's free padding (untiled offset 256), exactly as
    ``tools/test_c2_attn.py::c2_attn_resident_npu`` does it.  A single resident
    PDI coexists with the per-layer rgr PDI (which LoadPDI-resets c2 each layer),
    sidestepping the two-full-fabric-PDI wedge of the per-position build.
    """
    from kernel_builder.backend_presets import OGF_BACKEND
    from builders.c2_attn import (build_c2_attn_resident_module,
                                  c2_attn_resident_kernel_id)

    group_size = n_heads // n_kv_heads
    seq_len = current_pos + 1
    assert seq_len <= _RES_PADDED, (
        f"c2_attn resident supports seq_len<=256; got {seq_len} "
        f"(current_pos={current_pos})")
    TILE_ROWS = _RES_TILE_ROWS
    tile_size = TILE_ROWS * head_dim
    kv_size = _RES_MAX_CHUNKS * tile_size

    # ONE resident PDI for ALL positions/layers. compile_decode_kernels builds
    # it into the "o_gemv_ffn" cache slot under the resident XRT kernel id;
    # fall back to a lazy build under that same key if absent.
    kname = "o_gemv_ffn"
    if kname not in _C2_ATTN_BUILT:
        if kname not in cache.artifacts:
            ir = build_c2_attn_resident_module(n_kv_heads, verbose=False)
            cache.compile_and_cache(kname, ir, c2_attn_resident_kernel_id())
        _C2_ATTN_BUILT.add(kname)

    qh = np.asarray(q_roped, dtype=bfloat16).reshape(n_heads, head_dim)
    q_all = np.zeros(n_kv_heads * tile_size, dtype=bfloat16)
    k_all = np.zeros(n_kv_heads * kv_size, dtype=bfloat16)
    v_all = np.zeros(n_kv_heads * kv_size, dtype=bfloat16)
    for g in range(n_kv_heads):
        q_pad = np.zeros((TILE_ROWS, head_dim), dtype=bfloat16)
        q_pad[:group_size] = qh[g * group_size:(g + 1) * group_size]
        q_all[g * tile_size:(g + 1) * tile_size] = q_pad.reshape(-1)
        # KV zero-padded to 256 (fixed 4 chunks), PRE-TILED per 64-row chunk.
        k_pad = np.zeros((_RES_PADDED, head_dim), dtype=bfloat16)
        v_pad = np.zeros((_RES_PADDED, head_dim), dtype=bfloat16)
        k_pad[:seq_len] = k_cache_layer[g, :seq_len, :].astype(bfloat16)
        v_pad[:seq_len] = v_cache_layer[g, :seq_len, :].astype(bfloat16)
        for c in range(_RES_MAX_CHUNKS):
            kt = _c2_attn_tile_8x8(k_pad[c * TILE_ROWS:(c + 1) * TILE_ROWS])
            vt = _c2_attn_tile_8x8(v_pad[c * TILE_ROWS:(c + 1) * TILE_ROWS])
            base = g * kv_size + c * tile_size
            k_all[base:base + tile_size] = kt
            v_all[base:base + tile_size] = vt
        # Fold runtime valid length L = seq_len into q's free padding.  q_all is
        # UNTILED (the device DMA tiles it 8x8); the core reads tiled offset 32 =
        # (row4,col0), whose UNTILED position is row4*64+col0 = 256.  Rows 0..3
        # hold real heads (group_size<=4), so row 4 is free.
        q_all[g * tile_size + 256] = bfloat16(float(seq_len))

    wo = layer_weights._wo_t
    attn_scratch = np.zeros(n_kv_heads * tile_size, dtype=bfloat16)  # arg1 wide
    proj_buf = np.zeros(emb_dim, dtype=bfloat16)
    x_residual = x_bf16.flatten().astype(bfloat16)
    res1_buf = np.zeros(emb_dim, dtype=bfloat16)
    w_norm2 = layer_weights.ffn_norm.reshape(emb_dim).astype(bfloat16)
    normed2_buf = np.zeros(emb_dim, dtype=bfloat16)
    w_gate = layer_weights._wgate_t
    gate_buf = np.zeros(hidden_dim, dtype=bfloat16)
    w_up = layer_weights._wup_t
    up_buf = np.zeros(hidden_dim, dtype=bfloat16)
    swiglu_buf = np.zeros(hidden_dim, dtype=bfloat16)
    w_down = layer_weights._wdown_t
    down_buf = np.zeros(emb_dim, dtype=bfloat16)
    output_buf = np.zeros(emb_dim, dtype=bfloat16)

    layer_idx = getattr(layer_weights, "_layer_idx", None)
    bo_key = f"c2attn_{kname}_L{layer_idx}" if layer_idx is not None else None
    results = cache.load_and_run(
        kname, OGF_BACKEND,
        wo, attn_scratch, proj_buf, x_residual, res1_buf, w_norm2, normed2_buf,
        w_gate, gate_buf, w_up, up_buf, swiglu_buf, w_down, down_buf, output_buf,
        q_all, k_all, v_all,
        output_indices=[14],
        static_input_indices={0, 7, 9, 12},
        intermediate_indices={1, 2, 4, 6, 8, 10, 11, 13, 14},
        bo_key=bo_key,
    )
    return results[14].astype(bfloat16)


def _run_o_ffn_awq_experimental(
    cache,
    attn_out,
    x_bf16,
    layer_weights,
    awq_layer_weights,
    emb_dim,
    *,
    awq_tile_m_k2048,
    awq_tile_m_k8192,
):
    """Experimental direct-AWQ replacement for decode o_proj + FFN.

    Q/K/V still use the existing fused BF16 rms_gemv_rope path. This opt-in
    branch replaces the second fused BF16 decode kernel with four direct packed
    AWQ NPU GEMVs and CPU glue for add/RMSNorm/SwiGLU. It is correctness-first
    and intentionally separate from the default run-awq path. The direct AWQ
    GEMVs use the validated vecdeq kernel variant rather than the older scalar
    dequant loop.
    """
    from llama32_1b_awq_runtime import awq_gemv_npu_tiled
    from llama32_1b_reference import rms_norm

    def _tile_m(awq):
        return awq_tile_m_k8192 if awq.k == 8192 else awq_tile_m_k2048

    proj = awq_gemv_npu_tiled(
        cache,
        attn_out,
        awq_layer_weights.wo,
        tile_m=_tile_m(awq_layer_weights.wo),
        variant="vecdeq",
    )
    res1 = (proj.astype(np.float32) + x_bf16.flatten().astype(np.float32)).astype(bfloat16)

    normed2 = rms_norm(
        res1.astype(np.float32).reshape(1, emb_dim),
        layer_weights.ffn_norm.reshape(emb_dim).astype(np.float32),
    ).flatten().astype(bfloat16)

    gate = awq_gemv_npu_tiled(
        cache,
        normed2,
        awq_layer_weights.w_gate,
        tile_m=_tile_m(awq_layer_weights.w_gate),
        variant="vecdeq",
    )
    up = awq_gemv_npu_tiled(
        cache,
        normed2,
        awq_layer_weights.w_up,
        tile_m=_tile_m(awq_layer_weights.w_up),
        variant="vecdeq",
    )

    gate_f32 = gate.astype(np.float32)
    up_f32 = up.astype(np.float32)
    swiglu = (gate_f32 / (1.0 + np.exp(-gate_f32)) * up_f32).astype(bfloat16)

    down = awq_gemv_npu_tiled(
        cache,
        swiglu,
        awq_layer_weights.w_down,
        tile_m=_tile_m(awq_layer_weights.w_down),
        variant="vecdeq",
    )
    return (down.astype(np.float32) + res1.astype(np.float32)).astype(bfloat16)

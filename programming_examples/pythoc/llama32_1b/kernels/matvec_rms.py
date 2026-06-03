# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Packed RMSNorm prologue for air's 3-device o_gemv_ffn fold.

`rms_norm_packed_bf16` computes the post-attention RMSNorm into a resident
`normed` buffer, reading a PACKED `[2, K]` input (delivered as two chained
BDs on the gate/up tile's existing input channel):

    packed[0   : K  ] = res1         (pre-norm RMSNorm input)
    packed[K   : 2*K] = norm_weight   (ffn_norm_w, per-element)
    normed[i] = bf16(bf16(res1[i] * inv_rms) * norm_weight[i])

It is the prologue of the air "broadcast res1, RMSNorm per-tile" fusion, but
split out as its OWN kernel so the gate/up core can call it ONCE per token and
then loop the plain `matvec_vectorized_bf16_bf16` over the resident `normed`
across all output-row chunks -- amortizing the RMS over the whole tile instead
of recomputing it on every matvec call. The math/numeric sequence is
bit-identical to kernels/rms_norm.py (bf16 mean+eps, f32 rsqrt) so the HF
answer gate stays green.

Compiled to matvec_rms_pythoc.o via kernels/build.py:compile_matvec_rms.
"""

from aie.iron.pythoc import aie_kernel

from pythoc import bf16, f32, i32, ptr, void
from pythoc.aie import (
    aie_vector,
    broadcast,
    load_v,
    reduce_add_reassoc,
    store_v,
    vector_mul,
    zeros,
)

# bf16 MAC that accumulates bf16*bf16 products into an f32 accumulator -- the
# proven pythoc idiom for f32 accumulation (matvec.py uses the same op).
from pythoc.aie import I1024_I1024_ACC2048_bf_mac_conf  # noqa: F401

# Precise 1/sqrt via sqrtf (mirrors AIR math.rsqrt -> 1.0/llvm.intr.sqrt),
# not the AIE2P invsqrt hardware approximation.
from pythoc.aie import sqrtf  # noqa: F401


@aie_kernel
def rms_norm_packed_bf16(
    packed: ptr[bf16, True],   # [2*K]: [0:K]=res1, [K:2K]=norm_weight
    normed: ptr[bf16, True],   # [K] output normalized vector
    scratch: ptr[bf16, True],  # [16] horizontal-sum spill
) -> void:
    # Pass 1: sum of squares of res1 (packed[0:2048]) accumulated in F32 to
    # match the maintained AIR reference (bf16 square -> extf f32 -> f32 add).
    # `scratch` is unused now (f32 reduce replaces the scalar spill); kept for
    # ABI stability.
    acc: aie_vector[f32, 64] = zeros(f32, 64)
    conf: i32 = i32(60)
    p_x: ptr[bf16] = packed
    si: i32 = 0
    while si < 2048:
        xv: aie_vector[bf16, 64] = load_v(p_x, 64)
        acc = I1024_I1024_ACC2048_bf_mac_conf(xv, xv, acc, conf)  # sum sq -> f32
        p_x = p_x + 64
        si = si + 64

    total: f32 = reduce_add_reassoc(acc)                     # f32 horizontal reduce
    mean: f32 = total / f32(2048.0)                          # f32 divf
    mean_eps: f32 = mean + f32(1.0e-5)                       # f32, eps = 1e-5 (exact)
    inv_rms: bf16 = bf16(f32(1.0) / f32(sqrtf(mean_eps)))    # 1/sqrt precise, trunc bf16
    scale: aie_vector[bf16, 16] = broadcast(bf16, 16, inv_rms)

    # Pass 2: normed[i] = res1[i] * scale * norm_w[i]. norm_w is packed[2048:4096].
    p_x2: ptr[bf16] = packed
    p_w: ptr[bf16] = packed + 2048
    p_y: ptr[bf16] = normed
    nk: i32 = 0
    while nk < 2048:
        xv2: aie_vector[bf16, 16] = load_v(p_x2, 16)
        wv: aie_vector[bf16, 16] = load_v(p_w, 16)
        tmp: aie_vector[bf16, 16] = vector_mul(xv2, scale)
        out: aie_vector[bf16, 16] = vector_mul(tmp, wv)
        store_v(p_y, out)
        p_x2 = p_x2 + 16
        p_w = p_w + 16
        p_y = p_y + 16
        nk = nk + 16

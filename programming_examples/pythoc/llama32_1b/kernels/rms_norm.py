# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RMSNorm PythoC kernel for llama32_1b.

Replaces the AIR-generated inline RMSNorm body in
reference_mlir/{rms_gemv_rope,rms_gemms_rope}.npu.air.mlir. Math matches
HuggingFace Llama RMSNorm:

    mean_sq = sum(x[i]**2) / N
    y[i]    = x[i] / sqrt(mean_sq + eps) * w[i]

Computed in BF16 to match the AIR reference; only the final scalar
1/sqrt is promoted to f32 via the AIE2P invsqrt intrinsic.
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

# Lazy AIE2P scalar intrinsics. `sqrtf` is the precise HW sqrt; we form
# 1/sqrt(x) as 1.0/sqrtf(x) to mirror the AIR reference, whose math.rsqrt
# lowers to `1.0 / llvm.intr.sqrt` (a precise reciprocal-sqrt) -- NOT the
# AIE2P `invsqrt` hardware approximation. `build.py` mirrors these in
# extra_globals when invoking `compile_pythoc_source`.
from pythoc.aie import sqrtf  # noqa: F401


@aie_kernel
def rms_norm_2048_bf16(
    x: ptr[bf16, True],       # input vector, 2048 bf16
    w: ptr[bf16, True],       # weight vector, 2048 bf16
    y: ptr[bf16, True],       # output vector, 2048 bf16
    scratch: ptr[bf16, True], # 16 bf16 scratch for horizontal sum
) -> void:
    # Pass 1 -- sum of squares accumulated in F32 (matches the maintained AIR
    # reference matvec_swiglu_rms: bf16 square -> extf to f32 -> f32 add;
    # bf16 accumulation lost ~9% summing 2048 squared values). `scratch` is
    # unused now (the f32 horizontal reduce replaces the scalar spill) but
    # kept in the signature for ABI stability.
    acc: aie_vector[f32, 64] = zeros(f32, 64)
    conf: i32 = i32(60)
    p_x: ptr[bf16] = x
    i: i32 = 0
    while i < 2048:
        xv: aie_vector[bf16, 64] = load_v(p_x, 64)
        acc = I1024_I1024_ACC2048_bf_mac_conf(xv, xv, acc, conf)  # sum sq -> f32
        p_x = p_x + 64
        i = i + 64

    total: f32 = reduce_add_reassoc(acc)                     # f32 horizontal reduce
    mean: f32 = total / f32(2048.0)                          # f32 divf
    mean_eps: f32 = mean + f32(1.0e-5)                       # f32, eps = 1e-5 (exact)
    inv_rms: bf16 = bf16(f32(1.0) / f32(sqrtf(mean_eps)))    # 1/sqrt precise, trunc bf16
    scale: aie_vector[bf16, 16] = broadcast(bf16, 16, inv_rms)

    # Pass 2 -- y[i] = x[i] * scale * w[i]
    p_x2: ptr[bf16] = x
    p_w: ptr[bf16] = w
    p_y: ptr[bf16] = y
    k: i32 = 0
    while k < 2048:
        xv2: aie_vector[bf16, 16] = load_v(p_x2, 16)
        wv: aie_vector[bf16, 16] = load_v(p_w, 16)
        tmp: aie_vector[bf16, 16] = vector_mul(xv2, scale)
        out: aie_vector[bf16, 16] = vector_mul(tmp, wv)
        store_v(p_y, out)
        p_x2 = p_x2 + 16
        p_w = p_w + 16
        p_y = p_y + 16
        k = k + 16

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
    store_v,
    vector_add,
    vector_mul,
    zeros,
)

# Lazy AIE2P scalar intrinsics. `invsqrt` is resolved on first attribute
# access from pythoc.aie; this name binding lets PythoC's AST visitor see it
# as a user_global. `build.py` must mirror this in extra_globals when invoking
# `compile_pythoc_source` standalone (compile_pythoc_kernel ignores the source's
# imports).
from pythoc.aie import invsqrt  # noqa: F401


@aie_kernel
def rms_norm_2048_bf16(
    x: ptr[bf16, True],       # input vector, 2048 bf16
    w: ptr[bf16, True],       # weight vector, 2048 bf16
    y: ptr[bf16, True],       # output vector, 2048 bf16
    scratch: ptr[bf16, True], # 16 bf16 scratch for horizontal sum
) -> void:
    # Pass 1 -- accumulate squared values into a 16-wide BF16 vector
    accum: aie_vector[bf16, 16] = zeros(bf16, 16)
    p_x: ptr[bf16] = x
    i: i32 = 0
    while i < 2048:
        xv: aie_vector[bf16, 16] = load_v(p_x, 16)
        sq: aie_vector[bf16, 16] = vector_mul(xv, xv)
        accum = vector_add(accum, sq)
        p_x = p_x + 16
        i = i + 16

    # Horizontal sum: spill the 16-wide accumulator to scratch then scalar-sum.
    store_v(scratch, accum)
    s: bf16 = bf16(0.0)
    j: i32 = 0
    while j < 16:
        s = s + scratch[j]
        j = j + 1

    # mean_sq + eps, invsqrt, broadcast back to vector.
    # Match AIR-reference numerical sequence: bf16 mean+eps then f32 rsqrt.
    mean: bf16 = s * bf16(0.00048828125)  # 1/2048
    mean_eps: bf16 = mean + bf16(1.001360e-05)
    inv_rms: bf16 = bf16(invsqrt(f32(mean_eps)))
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

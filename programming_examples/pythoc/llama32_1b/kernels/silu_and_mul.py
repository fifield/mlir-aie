# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SwiGLU element-wise SiLU * up kernel for llama32_1b FFN.

Replaces `silu_and_mul.cc` (AIR-tree reference, hand-written C++ AIE API).
Math:

    SiLU(x) = x * sigmoid(x) = x * 0.5 * (tanh(x/2) + 1)
    out[i]  = SiLU(gate[i]) * up[i]

n is the per-call element count (FFN o_gemv_ffn / o_ffn callers pass 1024).
"""

from aie.iron.pythoc import aie_kernel

from pythoc import bf16, i32, ptr, void
from pythoc.aie import (
    aie_vector,
    broadcast,
    getTanhBf16,
    load_v,
    store_v,
    vector_add,
    vector_mul,
)


@aie_kernel
def silu_and_mul_bf16(
    gate: ptr[bf16, True],
    up: ptr[bf16, True],
    out: ptr[bf16, True],
    n: i32,
) -> void:
    vec: i32 = 16
    half: aie_vector[bf16, 16] = broadcast(bf16, 16, bf16(0.5))
    one: aie_vector[bf16, 16] = broadcast(bf16, 16, bf16(1.0))

    p_g: ptr[bf16] = gate
    p_u: ptr[bf16] = up
    p_o: ptr[bf16] = out

    i: i32 = 0
    while i < n:
        g: aie_vector[bf16, 16] = load_v(p_g, 16)
        u: aie_vector[bf16, 16] = load_v(p_u, 16)
        g_half: aie_vector[bf16, 16] = vector_mul(g, half)
        tanh_val: aie_vector[bf16, 16] = getTanhBf16(g_half)
        one_plus: aie_vector[bf16, 16] = vector_add(one, tanh_val)
        sigmoid: aie_vector[bf16, 16] = vector_mul(half, one_plus)
        silu: aie_vector[bf16, 16] = vector_mul(g, sigmoid)
        result: aie_vector[bf16, 16] = vector_mul(silu, u)
        store_v(p_o, result)
        p_g = p_g + vec
        p_u = p_u + vec
        p_o = p_o + vec
        i = i + vec

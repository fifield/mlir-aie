# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Half-split RoPE PythoC kernel (HuggingFace Llama convention).

Replaces `rope_halfsplit.cc`. Pairs (x[i], x[i + dims/2]) with rotation
angle theta_i. LUT layout: [cos_0..cos_{half-1}, sin_0..sin_{half-1}].

    out[i]        = x[i]      * cos[i] - x[i+half] * sin[i]
    out[i + half] = x[i]      * sin[i] + x[i+half] * cos[i]

Llama-3.2-1B head_dim is 64, so dims=64 and half=32. We process 16 lanes
per iteration (matches the .cc template parameter N=16). Loop fixed at
2 iterations (half / 16).
"""

from aie.iron.pythoc import aie_kernel

from pythoc import bf16, i32, ptr, void
from pythoc.aie import (
    aie_vector,
    load_v,
    store_v,
    vector_add,
    vector_mul,
    vector_sub,
)


@aie_kernel
def rope(
    input: ptr[bf16, True],
    lut: ptr[bf16, True],
    output: ptr[bf16, True],
    dims: i32,
) -> void:
    half: i32 = dims // 2
    vec: i32 = 16

    p_x1: ptr[bf16] = input
    p_x2: ptr[bf16] = input + half
    p_cos: ptr[bf16] = lut
    p_sin: ptr[bf16] = lut + half
    p_o1: ptr[bf16] = output
    p_o2: ptr[bf16] = output + half

    v: i32 = 0
    while v < half:
        x1: aie_vector[bf16, 16] = load_v(p_x1, 16)
        x2: aie_vector[bf16, 16] = load_v(p_x2, 16)
        cos_v: aie_vector[bf16, 16] = load_v(p_cos, 16)
        sin_v: aie_vector[bf16, 16] = load_v(p_sin, 16)

        x1_cos: aie_vector[bf16, 16] = vector_mul(x1, cos_v)
        x2_sin: aie_vector[bf16, 16] = vector_mul(x2, sin_v)
        out1: aie_vector[bf16, 16] = vector_sub(x1_cos, x2_sin)

        x1_sin: aie_vector[bf16, 16] = vector_mul(x1, sin_v)
        x2_cos: aie_vector[bf16, 16] = vector_mul(x2, cos_v)
        out2: aie_vector[bf16, 16] = vector_add(x1_sin, x2_cos)

        store_v(p_o1, out1)
        store_v(p_o2, out2)

        p_x1 = p_x1 + vec
        p_x2 = p_x2 + vec
        p_cos = p_cos + vec
        p_sin = p_sin + vec
        p_o1 = p_o1 + vec
        p_o2 = p_o2 + vec
        v = v + vec

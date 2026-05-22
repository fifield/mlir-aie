# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""BF16 matrix-vector multiply for K=8192 FFN down-projection (renamed symbols).

Replaces the AIR-tree reference `mv_k8192.o` which is the same `mv.cc`
source compiled with:
    -DDIM_M_OUTPUT=2
    -Dmatvec_vectorized_bf16_bf16=dg_matvec_vectorized_bf16_bf16
    -Dlinalg_fill_bf16=dg_linalg_fill_bf16

Used by `o_gemv_ffn` for the FFN down-projection (K=8192, m=1 per call,
output buffer is 2 bf16). Math is identical to kernels/matvec.py; the
only differences are the `dg_` symbol prefix and DIM_M_OUTPUT=2 (which
only affects the linalg_fill helper).
"""

from aie.iron.pythoc import aie_kernel

from pythoc import bf16, f32, i32, ptr, u32, void
from pythoc.aie import (
    aie_vector,
    load_v,
    reduce_add,
    zeros,
)

from pythoc.aie import I512_I512_ACC1024_bf_mac_conf  # noqa: F401


@aie_kernel
def dg_linalg_fill_bf16(c: ptr[bf16, True]) -> void:
    """Zero DIM_M_OUTPUT=2 bf16 elements at `c`.

    The .cc `zero_vectorized<bf16, 2, 1, 32>(c)` falls into the scalar
    remainder branch (vector store width 32 > buffer length 2). Mirror
    that with two scalar stores.

    Defined FIRST so compile_pythoc_source picks it up as a helper of
    `dg_matvec_vectorized_bf16_bf16` -- both symbols land in one .o.
    """
    c[0] = bf16(0.0)
    c[1] = bf16(0.0)


@aie_kernel
def dg_matvec_vectorized_bf16_bf16(
    m: u32,
    k: u32,
    row_offset: u32,
    a: ptr[bf16, True],
    b: ptr[bf16, True],
    c: ptr[bf16, True],
) -> void:
    r: u32 = u32(32)
    conf: i32 = i32(60)  # per-lane bf16 MAC; same as kernels/matvec.py

    p_c: ptr[bf16] = c + row_offset
    p_a_row: ptr[bf16] = a

    i: u32 = u32(0)
    while i < m:
        acc: aie_vector[f32, 32] = zeros(f32, 32)
        p_a: ptr[bf16] = p_a_row
        p_b: ptr[bf16] = b
        j: u32 = u32(0)
        while j < k:
            a_v: aie_vector[bf16, 32] = load_v(p_a, 32)
            b_v: aie_vector[bf16, 32] = load_v(p_b, 32)
            acc = I512_I512_ACC1024_bf_mac_conf(a_v, b_v, acc, conf)
            p_a = p_a + r
            p_b = p_b + r
            j = j + r

        s: f32 = reduce_add(acc)
        p_c[0] = bf16(s)

        p_c = p_c + u32(1)
        p_a_row = p_a_row + k
        i = i + u32(1)

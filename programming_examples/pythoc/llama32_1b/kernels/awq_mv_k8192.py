# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Packed-uint4 AWQ matvec for K=8192 FFN down-projection (renamed symbols).

Replaces the AIR-tree reference `awq_mv_k8192.o` which is the same
`awq_mv.cc` source compiled with:
    -DDIM_M_OUTPUT=2
    -DAWQ_MATVEC_FN=dg_awq_matvec_vectorized_u4_bf16
    -DAWQ_LINALG_FILL_FN=dg_awq_linalg_fill_bf16

Used by the fused o_gemv_ffn_awq decode kernel for the down-projection
(K=8192, m=1 per call, output buffer is 2 bf16). Math is identical to
kernels/awq_mv.py; the only differences are the `dg_` symbol prefix
and DIM_M_OUTPUT=2 (which only affects the linalg_fill helper).

Mirrors the kernels/matvec_k8192.py clone pattern (DIM_M_OUTPUT=2 instead
of 8 so the fill helper writes 2 bf16 elements instead of 8).
"""

from aie.iron.pythoc import aie_kernel

from pythoc import bf16, f32, i32, ptr, u8, u32, void

from pythoc.aie import set_ctrl_reg  # noqa: F401


GROUP_SIZE: i32 = 128
DIM_M_OUTPUT: i32 = 2  # K=8192 down-projection writes 2 bf16 per call


@aie_kernel
def dg_awq_linalg_fill_bf16(zero: bf16, c_out: ptr[bf16, True]) -> void:
    """Zero DIM_M_OUTPUT=2 bf16 elements at `c_out`.

    The .cc compiled with DIM_M_OUTPUT=2 writes 2 scalar stores
    (vector store width 32 > buffer length 2 falls into the scalar
    remainder branch). Mirror with two scalar stores.

    Defined FIRST so compile_pythoc_source picks it up as a helper of
    `dg_awq_matvec_vectorized_u4_bf16`; both symbols land in one .o.
    """
    c_out[0] = zero
    c_out[1] = zero


@aie_kernel
def dg_awq_matvec_vectorized_u4_bf16(
    m: u32,
    k: u32,
    row_offset: u32,
    combined_in: ptr[u8, True],
    x_in: ptr[bf16, True],
    c_out: ptr[bf16, True],
) -> void:
    """Same math as kernels/awq_mv.py::awq_matvec_vectorized_u4_bf16,
    only the symbol name differs (dg_* for down-projection).

    See kernels/awq_mv.py for the combined-row ABI details and the
    scalar-fallback rationale.
    """
    set_ctrl_reg(1, 12)

    groups: u32 = k / u32(GROUP_SIZE)
    packed_per_group: u32 = u32(GROUP_SIZE) / u32(2)
    packed_per_row: u32 = k / u32(2)
    params_bytes_per_row: u32 = u32(4) * groups
    row_stride_bytes: u32 = packed_per_row + params_bytes_per_row

    p_c: ptr[bf16] = c_out + row_offset

    row: u32 = u32(0)
    while row < m:
        acc: f32 = f32(0.0)

        row_base: ptr[u8] = combined_in + row * row_stride_bytes
        q_row: ptr[u8] = row_base
        p_row: ptr[bf16] = row_base + packed_per_row

        group: u32 = u32(0)
        while group < groups:
            scale: f32 = f32(p_row[0])
            zero: f32 = f32(p_row[1])

            x_group_offset: u32 = group * u32(GROUP_SIZE)
            q_group_offset: u32 = group * packed_per_group

            pair: u32 = u32(0)
            while pair < packed_per_group:
                packed: u8 = q_row[q_group_offset + pair]
                q_even: f32 = f32(packed & u8(15))
                q_odd: f32 = f32((packed >> u8(4)) & u8(15))

                w_even: f32 = (q_even - zero) * scale
                w_odd: f32 = (q_odd - zero) * scale

                x_even: f32 = f32(x_in[x_group_offset + u32(2) * pair])
                x_odd: f32 = f32(x_in[x_group_offset + u32(2) * pair + u32(1)])

                acc = acc + x_even * w_even
                acc = acc + x_odd * w_odd

                pair = pair + u32(1)

            p_row = p_row + u32(2)
            group = group + u32(1)

        p_c[row] = bf16(acc)
        row = row + u32(1)

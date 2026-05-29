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

Mirrors the kernels/matvec_k8192.py clone pattern.  Inner loop is the
**vectorized Fix2Float chain** (see kernels/awq_mv.py docstring for the
trick and the AIE-API origin).
"""

from aie.iron.pythoc import aie_kernel

from pythoc import bf16, f32, i32, i64, ptr, u8, u32, void

from pythoc.aie import (  # noqa: F401
    set_ctrl_reg,
    I1024_I1024_ACC2048_bf_mac_conf,
    I1024_I1024_ACC2048_bf_msc_conf,
    v32accfloat_to_v32bf16,
    v32bf16_to_v32accfloat,
    unpack_I1024_I8_I4,
    acc32_v32_I256_ups,
    ACC2048_add_conf,
    ACC2048_accfloat_sub_conf,
)
from pythoc.aie import (
    aie_vector,
    broadcast,
    concat,
    load_v,
    loop_range,
    prepare_for_pipelining,
    reduce_add_reassoc,
    vector_cast,
    vector_extract,
    vector_mul,
    zeros,
)


GROUP_SIZE: i32 = 128
DIM_M_OUTPUT: i32 = 2  # K=8192 down-projection writes 2 bf16 per call

# Fix2Float magic constants (see kernels/awq_mv.py docstring).
MAGIC_L_I32: i32
MAGIC_L_BF: bf16
CONF_BF16_MAC: i32


@aie_kernel
def dg_awq_linalg_fill_bf16(zero: bf16, c_out: ptr[bf16, True]) -> void:
    """Zero DIM_M_OUTPUT=2 bf16 elements at `c_out`."""
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
    """Same math/structure as kernels/awq_mv.py::awq_matvec_vectorized_u4_bf16,
    only the symbol name differs (``dg_*`` for the FFN down-projection).
    See kernels/awq_mv.py for the Fix2Float trick details.
    """
    set_ctrl_reg(1, 12)

    groups: u32 = k / u32(GROUP_SIZE)
    packed_per_group: u32 = u32(GROUP_SIZE) / u32(2)
    packed_per_row: u32 = k / u32(2)
    params_bytes_per_row: u32 = u32(4) * groups
    row_stride_bytes: u32 = packed_per_row + params_bytes_per_row

    chunks_per_group: u32 = packed_per_group / u32(32)

    # 64-lane Fix2Float magic constants (mirror clang aie::to_float; see
    # kernels/awq_mv.py for the rationale).
    magic_acc32_lanes: aie_vector[i32, 64] = broadcast(i32, 64, MAGIC_L_I32)
    magic_acc32_64: aie_vector[i64, 32] = vector_cast(magic_acc32_lanes, i64, 32)
    magic_bf: aie_vector[bf16, 32] = broadcast(bf16, 32, MAGIC_L_BF)
    magic_acc_32: aie_vector[f32, 32] = v32bf16_to_v32accfloat(magic_bf)
    magic_acc_64: aie_vector[f32, 64] = concat(magic_acc_32, magic_acc_32)

    p_c: ptr[bf16] = c_out + row_offset

    row: u32 = u32(0)
    while row < m:
        # Single 64-lane f32 accumulator; see kernels/awq_mv.py.
        acc: aie_vector[f32, 64] = zeros(f32, 64)

        row_base: ptr[u8] = combined_in + row * row_stride_bytes
        q_row: ptr[u8] = row_base
        p_row: ptr[bf16] = row_base + packed_per_row

        group: u32 = u32(0)
        # K=8192, GS=128 -> 64 groups per row.
        with prepare_for_pipelining():
            with loop_range(64):
                while group < groups:
                    scale_s: bf16 = p_row[0]
                    zero_s: bf16 = p_row[1]
                    zs_s: bf16 = scale_s * zero_s
                    scale_v: aie_vector[bf16, 32] = broadcast(bf16, 32, scale_s)
                    zs_v_64: aie_vector[bf16, 64] = broadcast(bf16, 64, zs_s)

                    x_group_offset: u32 = group * u32(GROUP_SIZE)
                    q_group_offset: u32 = group * packed_per_group

                    # Wide unpack: 64 packed bytes -> 128 nibbles per group
                    # (see kernels/awq_mv.py for the rationale).
                    q_packed: aie_vector[u8, 64] = load_v(q_row + q_group_offset, 64)
                    nibbles_all: aie_vector[u8, 128] = unpack_I1024_I8_I4(q_packed, i32(0))
                    nib_0: aie_vector[u8, 32] = vector_extract(nibbles_all, 0, 32)
                    nib_1: aie_vector[u8, 32] = vector_extract(nibbles_all, 32, 32)
                    nib_2: aie_vector[u8, 32] = vector_extract(nibbles_all, 64, 32)
                    nib_3: aie_vector[u8, 32] = vector_extract(nibbles_all, 96, 32)

                    # === Chunk 0 ===
                    # 64-lane Fix2Float front-end (mirrors clang aie::to_float).
                    lo_i32_0: aie_vector[i32, 32] = acc32_v32_I256_ups(nib_0, i32(0), i32(0))
                    hi_i32_0: aie_vector[i32, 32] = acc32_v32_I256_ups(nib_1, i32(0), i32(0))
                    comb_i32_0: aie_vector[i32, 64] = concat(lo_i32_0, hi_i32_0)
                    comb_i64_0: aie_vector[i64, 32] = vector_cast(comb_i32_0, i64, 32)
                    sum_i64_0: aie_vector[i64, 32] = ACC2048_add_conf(
                        comb_i64_0, magic_acc32_64, i32(0)
                    )
                    sum_acc_0: aie_vector[f32, 64] = vector_cast(sum_i64_0, f32, 64)
                    w_acc_0: aie_vector[f32, 64] = ACC2048_accfloat_sub_conf(
                        sum_acc_0, magic_acc_64, i32(60)
                    )
                    w_lo_acc_0: aie_vector[f32, 32] = vector_extract(w_acc_0, 0, 32)
                    w_hi_acc_0: aie_vector[f32, 32] = vector_extract(w_acc_0, 32, 32)
                    w_lo_bf_0: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(w_lo_acc_0)
                    w_hi_bf_0: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(w_hi_acc_0)
                    w_lo_s_0: aie_vector[bf16, 32] = vector_mul(w_lo_bf_0, scale_v)
                    w_hi_s_0: aie_vector[bf16, 32] = vector_mul(w_hi_bf_0, scale_v)
                    w_combined_0: aie_vector[bf16, 64] = concat(w_lo_s_0, w_hi_s_0)
                    x_combined_0: aie_vector[bf16, 64] = load_v(x_in + x_group_offset, 64)
                    acc = I1024_I1024_ACC2048_bf_mac_conf(
                        x_combined_0, w_combined_0, acc, CONF_BF16_MAC
                    )
                    acc = I1024_I1024_ACC2048_bf_msc_conf(
                        x_combined_0, zs_v_64, acc, CONF_BF16_MAC
                    )

                    # === Chunk 1 ===
                    lo_i32_1: aie_vector[i32, 32] = acc32_v32_I256_ups(nib_2, i32(0), i32(0))
                    hi_i32_1: aie_vector[i32, 32] = acc32_v32_I256_ups(nib_3, i32(0), i32(0))
                    comb_i32_1: aie_vector[i32, 64] = concat(lo_i32_1, hi_i32_1)
                    comb_i64_1: aie_vector[i64, 32] = vector_cast(comb_i32_1, i64, 32)
                    sum_i64_1: aie_vector[i64, 32] = ACC2048_add_conf(
                        comb_i64_1, magic_acc32_64, i32(0)
                    )
                    sum_acc_1: aie_vector[f32, 64] = vector_cast(sum_i64_1, f32, 64)
                    w_acc_1: aie_vector[f32, 64] = ACC2048_accfloat_sub_conf(
                        sum_acc_1, magic_acc_64, i32(60)
                    )
                    w_lo_acc_1: aie_vector[f32, 32] = vector_extract(w_acc_1, 0, 32)
                    w_hi_acc_1: aie_vector[f32, 32] = vector_extract(w_acc_1, 32, 32)
                    w_lo_bf_1: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(w_lo_acc_1)
                    w_hi_bf_1: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(w_hi_acc_1)
                    w_lo_s_1: aie_vector[bf16, 32] = vector_mul(w_lo_bf_1, scale_v)
                    w_hi_s_1: aie_vector[bf16, 32] = vector_mul(w_hi_bf_1, scale_v)
                    w_combined_1: aie_vector[bf16, 64] = concat(w_lo_s_1, w_hi_s_1)
                    x_combined_1: aie_vector[bf16, 64] = load_v(x_in + x_group_offset + u32(64), 64)
                    acc = I1024_I1024_ACC2048_bf_mac_conf(
                        x_combined_1, w_combined_1, acc, CONF_BF16_MAC
                    )
                    acc = I1024_I1024_ACC2048_bf_msc_conf(
                        x_combined_1, zs_v_64, acc, CONF_BF16_MAC
                    )

                    p_row = p_row + u32(2)
                    group = group + u32(1)

        s: f32 = reduce_add_reassoc(acc)
        p_c[row] = bf16(s)
        row = row + u32(1)

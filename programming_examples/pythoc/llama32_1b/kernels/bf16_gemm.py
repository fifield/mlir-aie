# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# BF16 × BF16 -> F32 GEMM tile kernel for AIE2P (NPU2).
#
# Adapted verbatim from
# `programming_examples/pythoc/bf16_gemm_multi_core.py::bf16_gemm_tile_kernel`.
# The standalone example builds this kernel + its own IRON program; we extract
# JUST the @aie_kernel function here so the placed-IRON builders for
# `rms_gemms_rope` and `o_ffn` can link it as an external `.o` and replace
# their current inline `vector.contract` ops with a single function call per
# core.
#
# All `_CONST` names below are compile-time scalars injected via
# `extra_globals=` at PythocKernel compile time (see kernels/build.py).
# Strides are passed explicitly so the kernel is layout-agnostic -- the same
# source builds different `.o` per (M_BLOCKS, N_BLOCKS, K_MICRO) configuration,
# and can be retargeted to either an `[M_BLOCKS, K_MICRO, 8, 8]` or
# `[K_MICRO, M_BLOCKS, 8, 8]` A layout just by swapping the A_*_STRIDE values.
#
# Reference layout (matches bf16_gemm_multi_core.py / `dims_to_stream` output):
#   A: [M_BLOCKS, K_MICRO, 8, 8]  -> A[m_blk, k_blk] at (m * K_MICRO + k) * 64
#   B: [K_MICRO, N_BLOCKS, 8, 8]  -> B[k_blk, n_blk] at (k * N_BLOCKS + n) * 64
#   C: [M_BLOCKS, N_BLOCKS, 8, 8] -> C[m_blk, n_blk] at (m * N_BLOCKS + n) * 64
#
# MAC: 2x2 register-blocked, BFP16 emulation (bf16 -> accfloat -> bfp16ebs8
# -> BFP576_BFP576_ACC2048_mac_conf). Output is f32 in C.

from pythoc import ptr, i32, f32, bf16, void
from pythoc.aie import (
    aie_vector,
    aie_kernel,
    concat,
    load_v,
    store_v,
    vector_cast,
    vector_extract,
    vshuffle,
    zeros,
    BFP576_BFP576_ACC2048_mac_conf,
    set_ctrl_reg,
    v32bf16_to_v32accfloat,
    v64accfloat_to_v64bfp16ebs8,
)
from pythoc.aie.profiling import event0, event1


@aie_kernel
def bf16_gemm_kernel(
    a_buf: ptr[bf16, True],
    b_buf: ptr[bf16, True],
    c_buf: ptr[f32, True],
) -> void:
    """Accumulate C += A × B for one (M_BLOCKS*8) × (K_MICRO*8) × (N_BLOCKS*8) tile.

    The C buffer is assumed pre-zeroed (or holding the running accumulator).
    Uses 2x2 register blocking so each pair of A rows is reused across a pair
    of B columns, halving the operand loads relative to a flat 1x1 schedule.
    """
    # Rounding-mode setup (matches the cached MLIR's emitted GEMM cores).
    # Reg 9 = conv round mode: 1 = round-to-nearest-even.
    # Reg 1 = MAC saturation/rounding config.
    set_ctrl_reg(9, 1)
    set_ctrl_reg(1, 12)
    event0()

    m: i32 = 0
    while m < M_BLOCKS_CONST:
        n: i32 = 0
        while n < N_BLOCKS_CONST:
            # 2x2 C tile offsets.
            c00_off: i32 = m * C_M_STRIDE_CONST + n * C_N_STRIDE_CONST
            c10_off: i32 = c00_off + C_M_STRIDE_CONST
            c01_off: i32 = c00_off + C_N_STRIDE_CONST
            c11_off: i32 = c00_off + C_M_STRIDE_CONST + C_N_STRIDE_CONST

            acc_c00: aie_vector[f32, 64] = load_v(c_buf + c00_off, 64)
            acc_c10: aie_vector[f32, 64] = load_v(c_buf + c10_off, 64)
            acc_c01: aie_vector[f32, 64] = load_v(c_buf + c01_off, 64)
            acc_c11: aie_vector[f32, 64] = load_v(c_buf + c11_off, 64)

            a0_off: i32 = m * A_M_STRIDE_CONST
            a1_off: i32 = a0_off + A_M_STRIDE_CONST
            b0_off: i32 = n * B_N_STRIDE_CONST
            b1_off: i32 = (n + 1) * B_N_STRIDE_CONST

            k: i32 = 0
            while k < K_MICRO_CONST:
                va0: aie_vector[bf16, 64] = load_v(a_buf + a0_off, 64)
                a0_off = a0_off + A_K_STRIDE_CONST
                vb0: aie_vector[bf16, 64] = load_v(b_buf + b0_off, 64)
                b0_off = b0_off + B_K_STRIDE_CONST

                # A0 -> bfp16ebs8 (no vshuffle).
                a0_lo: aie_vector[bf16, 32] = vector_extract(va0, 0, 32)
                a0_hi: aie_vector[bf16, 32] = vector_extract(va0, 32, 32)
                a0_acc_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a0_lo)
                a0_acc_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a0_hi)
                a0_acc: aie_vector[f32, 64] = concat(a0_acc_lo, a0_acc_hi)

                # B0 -> bfp16ebs8 (vshuffle path needed for column-major B).
                b0_i32: aie_vector[i32, 32] = vector_cast(vb0, i32, 32)
                b0_lo_i: aie_vector[i32, 16] = vector_extract(b0_i32, 0, 16)
                b0_hi_i: aie_vector[i32, 16] = vector_extract(b0_i32, 16, 16)
                b0_even: aie_vector[i32, 16] = vshuffle(b0_lo_i, b0_hi_i, 52)
                b0_odd: aie_vector[i32, 16] = vshuffle(b0_lo_i, b0_hi_i, 53)
                b0_cat: aie_vector[i32, 32] = concat(b0_even, b0_odd)
                vb0_s: aie_vector[bf16, 64] = vector_cast(b0_cat, bf16, 64)
                b0_s_lo: aie_vector[bf16, 32] = vector_extract(vb0_s, 0, 32)
                b0_s_hi: aie_vector[bf16, 32] = vector_extract(vb0_s, 32, 32)
                b0_acc_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(b0_s_lo)
                b0_acc_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(b0_s_hi)
                b0_acc: aie_vector[f32, 64] = concat(b0_acc_lo, b0_acc_hi)

                a0_mant, a0_exp = v64accfloat_to_v64bfp16ebs8(a0_acc)
                b0_mant, b0_exp = v64accfloat_to_v64bfp16ebs8(b0_acc)

                # C00 += A0 × B0
                acc_i00: aie_vector[i32, 64] = vector_cast(acc_c00, i32, 64)
                res00: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
                    a0_mant, a0_exp, b0_mant, b0_exp, acc_i00, MAC_CONF
                )

                # B1 -> bfp16ebs8
                vb1: aie_vector[bf16, 64] = load_v(b_buf + b1_off, 64)
                b1_off = b1_off + B_K_STRIDE_CONST

                b1_i32: aie_vector[i32, 32] = vector_cast(vb1, i32, 32)
                b1_lo_i: aie_vector[i32, 16] = vector_extract(b1_i32, 0, 16)
                b1_hi_i: aie_vector[i32, 16] = vector_extract(b1_i32, 16, 16)
                b1_even: aie_vector[i32, 16] = vshuffle(b1_lo_i, b1_hi_i, 52)
                b1_odd: aie_vector[i32, 16] = vshuffle(b1_lo_i, b1_hi_i, 53)
                b1_cat: aie_vector[i32, 32] = concat(b1_even, b1_odd)
                vb1_s: aie_vector[bf16, 64] = vector_cast(b1_cat, bf16, 64)
                b1_s_lo: aie_vector[bf16, 32] = vector_extract(vb1_s, 0, 32)
                b1_s_hi: aie_vector[bf16, 32] = vector_extract(vb1_s, 32, 32)
                b1_acc_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(b1_s_lo)
                b1_acc_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(b1_s_hi)
                b1_acc: aie_vector[f32, 64] = concat(b1_acc_lo, b1_acc_hi)
                b1_mant, b1_exp = v64accfloat_to_v64bfp16ebs8(b1_acc)

                # C01 += A0 × B1   (A0 reused)
                acc_i01: aie_vector[i32, 64] = vector_cast(acc_c01, i32, 64)
                res01: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
                    a0_mant, a0_exp, b1_mant, b1_exp, acc_i01, MAC_CONF
                )

                # A1 -> bfp16ebs8 (no vshuffle).
                va1: aie_vector[bf16, 64] = load_v(a_buf + a1_off, 64)
                a1_off = a1_off + A_K_STRIDE_CONST

                a1_lo: aie_vector[bf16, 32] = vector_extract(va1, 0, 32)
                a1_hi: aie_vector[bf16, 32] = vector_extract(va1, 32, 32)
                a1_acc_lo: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a1_lo)
                a1_acc_hi: aie_vector[f32, 32] = v32bf16_to_v32accfloat(a1_hi)
                a1_acc: aie_vector[f32, 64] = concat(a1_acc_lo, a1_acc_hi)
                a1_mant, a1_exp = v64accfloat_to_v64bfp16ebs8(a1_acc)

                # C10 += A1 × B0   (B0 reused, then dies)
                acc_i10: aie_vector[i32, 64] = vector_cast(acc_c10, i32, 64)
                res10: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
                    a1_mant, a1_exp, b0_mant, b0_exp, acc_i10, MAC_CONF
                )

                # C11 += A1 × B1   (B1 reused, then dies)
                acc_i11: aie_vector[i32, 64] = vector_cast(acc_c11, i32, 64)
                res11: aie_vector[i32, 64] = BFP576_BFP576_ACC2048_mac_conf(
                    a1_mant, a1_exp, b1_mant, b1_exp, acc_i11, MAC_CONF
                )

                acc_c00 = vector_cast(res00, f32, 64)
                acc_c10 = vector_cast(res10, f32, 64)
                acc_c01 = vector_cast(res01, f32, 64)
                acc_c11 = vector_cast(res11, f32, 64)

                k = k + 1

            store_v(c_buf + c00_off, acc_c00)
            store_v(c_buf + c10_off, acc_c10)
            store_v(c_buf + c01_off, acc_c01)
            store_v(c_buf + c11_off, acc_c11)

            n = n + 2
        m = m + 2

    event1()

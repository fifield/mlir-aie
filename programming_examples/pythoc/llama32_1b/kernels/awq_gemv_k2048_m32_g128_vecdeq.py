# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Packed-uint4 AWQ GEMV, dim-specialized for K=2048, M=32, GROUP_SIZE=128.

Replaces the AIR-tree reference ``awq_gemv_k2048_m32_g128_vecdeq.o`` which
is compiled from awq_gemv.cc with:
    -DAWQ_GEMV_K=2048 -DAWQ_GEMV_M=32 -DAWQ_GEMV_GROUP_SIZE=128
    -DAWQ_GEMV_VECTORIZE_INLINE_DEQUANT=1

For each row 0..M-1:
    y[row] = sum_group [ sum_pair (
        x[g*GS + 2*pair]     * ((q_lo[pair] - zero) * scale) +
        x[g*GS + 2*pair + 1] * ((q_hi[pair] - zero) * scale)
    ) ]

Unlike the fused awq_mv kernel, the standalone GEMV takes ``qweight`` and
``params`` as SEPARATE buffers (not combined).

Stage 2 uses the SCALAR per-nibble fallback (mirrors awq_gemv.cc:154-161
``#else`` branch). The "vecdeq" name preserves the on-disk filename and
the cached MLIR's ``link_with`` string convention; the inner-loop math is
correctness-first scalar, with vectorization deferred to a later phase.
"""

from aie.iron.pythoc import aie_kernel

from pythoc import bf16, f32, i32, ptr, u8, u32, void

from pythoc.aie import set_ctrl_reg  # noqa: F401


# Compile-time dimensions baked in (mirrors C++ macros).
K: i32 = 2048
M: i32 = 32
GROUP_SIZE: i32 = 128


@aie_kernel
def awq_gemv_u4_bf16(
    x: ptr[bf16, True],
    qweight: ptr[u8, True],
    params: ptr[bf16, True],
    y: ptr[bf16, True],
) -> void:
    """Dim-specialized packed-uint4 AWQ GEMV (K=2048, M=32, GS=128)."""
    # Round-to-even: matches aie::set_rounding(conv_even) in awq_gemv.cc:129.
    set_ctrl_reg(1, 12)

    groups: u32 = u32(K) / u32(GROUP_SIZE)
    packed_per_group: u32 = u32(GROUP_SIZE) / u32(2)
    packed_per_row: u32 = u32(K) / u32(2)
    params_per_row: u32 = u32(2) * groups   # bf16 elements per row (scale,zero pairs)

    row: u32 = u32(0)
    while row < u32(M):
        acc: f32 = f32(0.0)
        q_row: ptr[u8] = qweight + row * packed_per_row
        p_row: ptr[bf16] = params + row * params_per_row

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

                x_even: f32 = f32(x[x_group_offset + u32(2) * pair])
                x_odd: f32 = f32(x[x_group_offset + u32(2) * pair + u32(1)])

                acc = acc + x_even * w_even
                acc = acc + x_odd * w_odd

                pair = pair + u32(1)

            p_row = p_row + u32(2)
            group = group + u32(1)

        y[row] = bf16(acc)
        row = row + u32(1)

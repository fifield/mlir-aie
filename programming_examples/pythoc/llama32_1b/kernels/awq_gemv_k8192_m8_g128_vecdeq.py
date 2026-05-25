# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Packed-uint4 AWQ GEMV, dim-specialized for K=8192, M=8, GROUP_SIZE=128.

Replaces the AIR-tree reference ``awq_gemv_k8192_m8_g128_vecdeq.o``.

Same code shape as awq_gemv_k2048_m32_g128_vecdeq.py with K=8192, M=8.
The symbol name (``awq_gemv_u4_bf16``) is identical; the standalone path
is dim-specialized at the launch level (each cached MLIR points at its
own ``.o``), so K=2048 and K=8192 variants live in separate ELFs.

See kernels/awq_gemv_k2048_m32_g128_vecdeq.py for the algorithm details
and rationale.
"""

from aie.iron.pythoc import aie_kernel

from pythoc import bf16, f32, i32, ptr, u8, u32, void

from pythoc.aie import set_ctrl_reg  # noqa: F401


K: i32 = 8192
M: i32 = 8
GROUP_SIZE: i32 = 128


@aie_kernel
def awq_gemv_u4_bf16(
    x: ptr[bf16, True],
    qweight: ptr[u8, True],
    params: ptr[bf16, True],
    y: ptr[bf16, True],
) -> void:
    """Dim-specialized packed-uint4 AWQ GEMV (K=8192, M=8, GS=128)."""
    set_ctrl_reg(1, 12)

    groups: u32 = u32(K) / u32(GROUP_SIZE)
    packed_per_group: u32 = u32(GROUP_SIZE) / u32(2)
    packed_per_row: u32 = u32(K) / u32(2)
    params_per_row: u32 = u32(2) * groups

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

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Packed-uint4 AWQ matrix-vector multiply with combined-row ABI.

Replaces the AIR-tree reference `awq_mv.cc` (Stage 1 C++ scaffolding).
For each output row:
    c[i + row_offset] = sum_group [ sum_pair (
        x[g*GROUP_SIZE + 2*pair]     * ((q_lo[pair] - zero) * scale) +
        x[g*GROUP_SIZE + 2*pair + 1] * ((q_hi[pair] - zero) * scale)
    ) ]
where each packed `q` byte holds two uint4 nibbles (low=even K, high=odd K)
per the AWQ packing convention, and per-group scale/zero come from the
combined buffer's params section.

Combined ABI: ``combined_in`` is row-major ``uint8[m, k/2 + 4*groups]``
where each row is laid out as ``[qweight (k/2 bytes)] [params (2*groups bf16
= 4*groups bytes)]``. The params section interleaves [scale, zero] pairs
per group.

Stage 2 uses the SCALAR per-nibble fallback (mirrors awq_mv.cc:106-111
``#else`` branch). The vectorized ``aie::vector_cast<uint4> +
aie::to_float<bfloat16>`` chain requires a PythoC bitcast op that doesn't
exist yet; vectorization is deferred to a later phase.

``set_ctrl_reg(1, 12)`` at entry mirrors ``aie::set_rounding(conv_even)``
in awq_mv.cc:130 (Stage 0 verified register 1 = crRnd, value 12 =
rnd_conv_even from llvm-aie's aie2p_defines.h).
"""

from aie.iron.pythoc import aie_kernel

from pythoc import bf16, f32, i32, ptr, u8, u32, void

# Lazy intrinsic for round-to-even rounding mode.
from pythoc.aie import set_ctrl_reg  # noqa: F401


# Compile-time constants (mirror awq_mv.cc:35,43,47,51 macros).
GROUP_SIZE: i32 = 128       # AWQ_MV_GROUP_SIZE: nibbles per quant group
DIM_M_OUTPUT: i32 = 8       # Output rows for the zero-fill helper (tile_m=8 path)


@aie_kernel
def awq_linalg_fill_bf16(zero: bf16, c_out: ptr[bf16, True]) -> void:
    """Zero DIM_M_OUTPUT bf16 elements at `c_out` (mirrors awq_mv.cc:157-161).

    Defined FIRST in the source so when compile_pythoc_source runs with
    function_name="awq_matvec_vectorized_u4_bf16", this becomes a "helper"
    and gets compiled into the same .o (single awq_mv_pythoc.o carries
    both symbols). Helpers that come AFTER the named entry point in
    source order are skipped by the AST walker.
    """
    i: u32 = u32(0)
    while i < u32(DIM_M_OUTPUT):
        c_out[i] = zero
        i = i + u32(1)


@aie_kernel
def awq_matvec_vectorized_u4_bf16(
    m: u32,
    k: u32,
    row_offset: u32,
    combined_in: ptr[u8, True],
    x_in: ptr[bf16, True],
    c_out: ptr[bf16, True],
) -> void:
    """Packed-uint4 AWQ matvec with combined-row ABI (runtime m/k).

    Mirrors awq_mv.cc:126-155 ``#else`` branch (scalar fallback). Row stride
    math from awq_mv.cc:133-144:
      groups              = k / GROUP_SIZE
      packed_per_group    = GROUP_SIZE / 2          (uint4 -> 2 nibbles/byte)
      packed_per_row      = k / 2
      params_bytes_per_row= 4 * groups              (2*groups bf16 == 4 bytes)
      row_stride_bytes    = packed_per_row + params_bytes_per_row
    """
    # Round-to-even for bf16 stores (matches aie::set_rounding(conv_even)
    # in awq_mv.cc:130).  Register 1 = crRnd; value 12 = rnd_conv_even per
    # llvm-aie aie2p_defines.h. attn.py uses the same constant.
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

        # Pointer arithmetic on a u8 ptr uses 1-byte stride, so row_base
        # is in bytes. q_row is the qweight section start; p_row is the
        # bf16 params section start -- ptr_to_ptr assignment emits an
        # LLVM bitcast (PythoC type_converter._convert_ptr_to_ptr) so we
        # can read bf16 elements directly from u8 storage.
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
                # Low nibble = even K index, high nibble = odd K index
                # (AWQ pack convention, awq_mv.cc:108-109).
                q_even: f32 = f32(packed & u8(15))
                q_odd: f32 = f32((packed >> u8(4)) & u8(15))

                w_even: f32 = (q_even - zero) * scale
                w_odd: f32 = (q_odd - zero) * scale

                x_even: f32 = f32(x_in[x_group_offset + u32(2) * pair])
                x_odd: f32 = f32(x_in[x_group_offset + u32(2) * pair + u32(1)])

                acc = acc + x_even * w_even
                acc = acc + x_odd * w_odd

                pair = pair + u32(1)

            # Each group consumes 2 bf16 params (scale, zero).
            p_row = p_row + u32(2)
            group = group + u32(1)

        p_c[row] = bf16(acc)
        row = row + u32(1)

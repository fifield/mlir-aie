# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Packed-uint4 AWQ matrix-vector multiply with combined-row ABI.

Replaces the AIR-tree reference `awq_mv.cc`.  For each output row:
    c[i + row_offset] = sum_group [ sum_pair (
        x[g*GROUP_SIZE + 2*pair]     * ((q_lo[pair] - zero) * scale) +
        x[g*GROUP_SIZE + 2*pair + 1] * ((q_hi[pair] - zero) * scale)
    ) ]
where each packed `q` byte holds two uint4 nibbles (low=even K, high=odd K)
per the AWQ packing convention, and per-group scale/zero come from the
combined buffer's params section.

Combined ABI: ``combined_in`` is row-major ``uint8[m, k/2 + 4*groups]``
where each row is laid out as ``[qweight (k/2 bytes)] [params (2*groups bf16
= 4*groups bytes)]``.  The params section interleaves [scale, zero] pairs
per group.

**Vectorized via Fix2Float** (mirrors awq_mv.cc:67-119 ``#if`` branch).
The uint4 -> bf16 conversion uses the AIE-API magic-number reinterpret
trick (aie_api/detail/aie2p/elementary.hpp:51-58):

  1. Zero-extend u8 nibbles to ``<32 x i32>`` via UPS chain (u8->i16->i32).
  2. Integer-add the magic constant ``0x4b010000`` per lane.
  3. Bitcast ``<32 x i32>`` to ``<32 x f32>`` (= accfloat representation).
  4. Subtract magic as accfloat via ``bf_msc_conf(magic_bf, ones_bf, acc)``
     -- folds the float-subtract into a hardware MSC (multiply-subtract).
  5. ``v32accfloat_to_v32bf16`` produces the final ``<32 x bf16>``.

Vector ``fadd <32 x f32>`` and ``fsub <32 x f32>`` do not legalize on AIE2P
GISel, so step 4 uses the bf16 MSC instead of plain fsub.  Similarly
``uitofp <N x iX> to <N x bfloat>`` does not legalize, so we route through
integer-add + bitcast.

``set_ctrl_reg(1, 12)`` mirrors ``aie::set_rounding(conv_even)``: register
1 = crRnd, value 12 = rnd_conv_even (Stage 0 verified via llvm-aie
``aie2p_defines.h``).  conf=60 selects per-lane bf16 MAC mode on the MSC
intrinsic (matches kernels/matvec.py:73; conf=0 silently breaks).
"""

from aie.iron.pythoc import aie_kernel

from pythoc import bf16, f32, i16, i32, ptr, u8, u32, void

# Lazy intrinsics.  ``set_ctrl_reg`` selects rounding mode at the start of
# each entry; ``I512_I512_ACC1024_bf_msc_conf`` is the 32-lane bf16 MSC
# used by the Fix2Float chain and by the actual MAC against x.
from pythoc.aie import (  # noqa: F401
    set_ctrl_reg,
    I512_I512_ACC1024_bf_msc_conf,
    I512_I512_ACC1024_bf_mac_conf,
    v32accfloat_to_v32bf16,
    unpack_I512_I8_I4,
)
from pythoc.aie import (
    aie_vector,
    broadcast,
    load_v,
    unpack_unsigned,
    vector_add,
    vector_cast,
    vector_extract,
    vector_mul,
    vector_sub,
    zeros,
)


# Compile-time constants (mirror awq_mv.cc:35,43,47,51 macros).
GROUP_SIZE: i32 = 128       # AWQ_MV_GROUP_SIZE: nibbles per quant group
DIM_M_OUTPUT: i32 = 8       # Output rows for the zero-fill helper (tile_m=8 path)

# Fix2Float magic constants (mirror aie_api/detail/aie2p/elementary.hpp:34).
# 0x4b01 as bf16 == 8454144.0 as f32 == 0x4b010000 in i32 bit pattern.
# `MAGIC_L_I32` is the per-lane integer constant added in acc32 space.
# `MAGIC_L_BF` is the same constant interpreted as bf16, used in the MSC
# subtract via `magic_l * 1.0 = magic_l`.
MAGIC_L_I32: i32        # = i32(0x4b010000)
MAGIC_L_BF: bf16        # = bf16(8454144.0)
CONF_BF16_MAC: i32      # = i32(60)


@aie_kernel
def awq_linalg_fill_bf16(zero: bf16, c_out: ptr[bf16, True]) -> void:
    """Zero DIM_M_OUTPUT bf16 elements at `c_out` (mirrors awq_mv.cc:157-161).

    Defined FIRST in the source so when compile_pythoc_source runs with
    function_name="awq_matvec_vectorized_u4_bf16", this becomes a "helper"
    and gets compiled into the same .o (single awq_mv_pythoc.o carries
    both symbols).  Helpers that come AFTER the named entry point in
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

    Vectorized inner loop:
      * Outer: per-row, per-group
      * Inner per group: load 32 packed bytes -> unpack to 64 nibbles ->
        Fix2Float each 32-lane half to bf16 -> apply (w-zero)*scale ->
        MAC against x_vec.

    For GROUP_SIZE=128 (default), one group is 64 packed bytes = 2 outer
    chunks of 32 bytes each.
    """
    # Round-to-even for bf16 stores.
    set_ctrl_reg(1, 12)

    groups: u32 = k / u32(GROUP_SIZE)
    packed_per_group: u32 = u32(GROUP_SIZE) / u32(2)   # 64 for GS=128
    packed_per_row: u32 = k / u32(2)
    params_bytes_per_row: u32 = u32(4) * groups
    row_stride_bytes: u32 = packed_per_row + params_bytes_per_row

    # Number of 32-byte chunks per group (= number of 64-nibble blocks).
    chunks_per_group: u32 = packed_per_group / u32(32)  # 2 for GS=128

    # Hoist magic constants out of the hot loop.
    magic_acc32: aie_vector[i32, 32] = broadcast(i32, 32, MAGIC_L_I32)
    magic_bf: aie_vector[bf16, 32] = broadcast(bf16, 32, MAGIC_L_BF)
    ones_bf: aie_vector[bf16, 32] = broadcast(bf16, 32, bf16(1.0))

    p_c: ptr[bf16] = c_out + row_offset

    row: u32 = u32(0)
    while row < m:
        # Per-row accumulator -- two 32-lane accfloat halves combined at the
        # end via reduce_add to give a single scalar dot product.
        acc_lo: aie_vector[f32, 32] = zeros(f32, 32)
        acc_hi: aie_vector[f32, 32] = zeros(f32, 32)

        row_base: ptr[u8] = combined_in + row * row_stride_bytes
        q_row: ptr[u8] = row_base
        p_row: ptr[bf16] = row_base + packed_per_row

        group: u32 = u32(0)
        while group < groups:
            scale_s: bf16 = p_row[0]
            zero_s: bf16 = p_row[1]
            scale_v: aie_vector[bf16, 32] = broadcast(bf16, 32, scale_s)
            zero_v: aie_vector[bf16, 32] = broadcast(bf16, 32, zero_s)

            x_group_offset: u32 = group * u32(GROUP_SIZE)
            q_group_offset: u32 = group * packed_per_group

            chunk: u32 = u32(0)
            while chunk < chunks_per_group:
                # 32 packed bytes per chunk = 64 nibbles after unpack.
                q_chunk: aie_vector[u8, 32] = load_v(
                    q_row + q_group_offset + chunk * u32(32), 32
                )

                # uint4 nibble unpack: <32 x u8> packed -> <64 x u8> nibbles.
                # Each output lane holds a single nibble value 0..15.  After
                # unpack the interleave is [low_nib_byte0, high_nib_byte0,
                # low_nib_byte1, high_nib_byte1, ...] which matches the AWQ
                # packing convention (low=even K, high=odd K) -- so output
                # lane k corresponds directly to K[k].
                nibbles: aie_vector[u8, 64] = unpack_I512_I8_I4(q_chunk, i32(0))

                # Split into two 32-lane halves so the rest of the chain
                # operates at 32 lanes (where ACC1024 MSC + accfloat->bf16
                # variants exist).  vector_extract(vec, start, count).
                nib_lo: aie_vector[u8, 32] = vector_extract(nibbles, 0, 32)
                nib_hi: aie_vector[u8, 32] = vector_extract(nibbles, 32, 32)

                # UPS widening chain (AIE2P GISel can't G_ZEXT vectors):
                #   <32 x u8> --unpack_I16_I8--> <32 x i16>
                #   <32 x i16> --acc32.v32.I512.ups--> <32 x i32>
                lo_i16: aie_vector[i16, 32] = unpack_unsigned(nib_lo, i16)
                hi_i16: aie_vector[i16, 32] = unpack_unsigned(nib_hi, i16)
                lo_i32: aie_vector[i32, 32] = unpack_unsigned(lo_i16, i32)
                hi_i32: aie_vector[i32, 32] = unpack_unsigned(hi_i16, i32)

                # Fix2Float on each half (LOW lanes).
                sum_lo_i32: aie_vector[i32, 32] = vector_add(lo_i32, magic_acc32)
                sum_lo_acc: aie_vector[f32, 32] = vector_cast(sum_lo_i32, f32, 32)
                # acc' = sum_acc - magic_bf * 1.0 (MSC computes acc - a*b).
                w_lo_acc: aie_vector[f32, 32] = I512_I512_ACC1024_bf_msc_conf(
                    magic_bf, ones_bf, sum_lo_acc, CONF_BF16_MAC
                )
                w_lo_bf: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(w_lo_acc)

                # Fix2Float on the HIGH 32 lanes.
                sum_hi_i32: aie_vector[i32, 32] = vector_add(hi_i32, magic_acc32)
                sum_hi_acc: aie_vector[f32, 32] = vector_cast(sum_hi_i32, f32, 32)
                w_hi_acc: aie_vector[f32, 32] = I512_I512_ACC1024_bf_msc_conf(
                    magic_bf, ones_bf, sum_hi_acc, CONF_BF16_MAC
                )
                w_hi_bf: aie_vector[bf16, 32] = v32accfloat_to_v32bf16(w_hi_acc)

                # AWQ dequant: w = (nibble_as_bf16 - zero) * scale.
                w_lo_dq: aie_vector[bf16, 32] = vector_mul(
                    vector_sub(w_lo_bf, zero_v), scale_v
                )
                w_hi_dq: aie_vector[bf16, 32] = vector_mul(
                    vector_sub(w_hi_bf, zero_v), scale_v
                )

                # MAC against x.  x[k] pairs with weight at K=k.
                # 64 K-lanes per chunk: 32 even + 32 odd in the original
                # nibble interleave, but after `vector_extract` halves we
                # have nib_lo = lanes 0..31 (K_offset+0..K_offset+31) and
                # nib_hi = lanes 32..63 (K_offset+32..K_offset+63).
                x_lo: aie_vector[bf16, 32] = load_v(
                    x_in + x_group_offset + chunk * u32(64), 32
                )
                x_hi: aie_vector[bf16, 32] = load_v(
                    x_in + x_group_offset + chunk * u32(64) + u32(32), 32
                )

                acc_lo = I512_I512_ACC1024_bf_mac_conf(
                    x_lo, w_lo_dq, acc_lo, CONF_BF16_MAC
                )
                acc_hi = I512_I512_ACC1024_bf_mac_conf(
                    x_hi, w_hi_dq, acc_hi, CONF_BF16_MAC
                )

                chunk = chunk + u32(1)

            p_row = p_row + u32(2)
            group = group + u32(1)

        # Reduce the two 32-lane accfloat accumulators to a single scalar.
        from pythoc.aie import reduce_add  # noqa: F401
        s_lo: f32 = reduce_add(acc_lo)
        s_hi: f32 = reduce_add(acc_hi)
        p_c[row] = bf16(s_lo + s_hi)
        row = row + u32(1)

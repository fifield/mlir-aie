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

from pythoc import bf16, f32, i32, i64, ptr, u8, u32, void

# Lazy intrinsics.  ``set_ctrl_reg`` selects rounding mode at the start of
# each entry. The Fix2Float chain stays at 32-lane (no v64accfloat_to_v64bf16
# exists in pythoc), but the final MAC/MSC against x widens to 64-lane via
# the I1024 intrinsics: concat the lo/hi dequantized halves into a single
# bf16[64] and accumulate into bf16[64] f32 accumulator. Doubles MAC
# throughput per group.
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

    # Hoist magic constants out of the hot loop.  These mirror clang's
    # aie::to_float<bfloat16> 64-lane Fix2Float front-end:
    #   * magic_acc32_64: integer magic 0x4b010000 broadcast to 64 acc32
    #     lanes, repacked as <32 x i64> for ACC2048.add.conf.
    #   * magic_acc_64: bf16 magic 0x4b01 routed through
    #     v32bf16_to_v32accfloat then concatenated to 64 accfloat lanes;
    #     ACC2048.accfloat.sub.conf subtracts it to undo the integer bias.
    magic_acc32_lanes: aie_vector[i32, 64] = broadcast(i32, 64, MAGIC_L_I32)
    magic_acc32_64: aie_vector[i64, 32] = vector_cast(magic_acc32_lanes, i64, 32)
    magic_bf: aie_vector[bf16, 32] = broadcast(bf16, 32, MAGIC_L_BF)
    magic_acc_32: aie_vector[f32, 32] = v32bf16_to_v32accfloat(magic_bf)
    magic_acc_64: aie_vector[f32, 64] = concat(magic_acc_32, magic_acc_32)

    p_c: ptr[bf16] = c_out + row_offset

    row: u32 = u32(0)
    while row < m:
        # Single 64-lane f32 accumulator covering K=0..63 of a chunk; the
        # two chunks per group both feed this one acc so register pressure
        # stays low. Each chunk turns into 1 wide MAC + 1 wide MSC against
        # x (was 4 narrow ops), doubling K-throughput.
        acc: aie_vector[f32, 64] = zeros(f32, 64)

        row_base: ptr[u8] = combined_in + row * row_stride_bytes
        q_row: ptr[u8] = row_base
        p_row: ptr[bf16] = row_base + packed_per_row

        group: u32 = u32(0)
        # K/GROUP_SIZE groups per row (16 for K=2048). Loop hints unlock
        # peano's zero-overhead hardware loop + software pipelining; see
        # kernels/matvec.py for the rationale.
        with prepare_for_pipelining():
            with loop_range(16):
                while group < groups:
                    scale_s: bf16 = p_row[0]
                    zero_s: bf16 = p_row[1]
                    # Math fusion: replace (w_bf - zero) * scale (vsub + vmul,
                    # with an accfloat<->bf16 round trip) with a fused MAC+MSC
                    # pair.  Precompute zs = zero * scale (scalar per group)
                    # and broadcast.  Then:
                    #   acc += x * (w_bf - z) * s
                    #        = x * (w_bf * s - zs)
                    # We still need w_scaled = w_bf * s in bf16 so the MAC has
                    # bf16 inputs; that's 1 bf16 mul per chunk-half.  The
                    # remaining MSC uses x and zs_v directly without further
                    # conversions.
                    zs_s: bf16 = scale_s * zero_s
                    scale_v: aie_vector[bf16, 32] = broadcast(bf16, 32, scale_s)
                    zs_v_lo: aie_vector[bf16, 32] = broadcast(bf16, 32, zs_s)
                    zs_v_64: aie_vector[bf16, 64] = broadcast(bf16, 64, zs_s)

                    x_group_offset: u32 = group * u32(GROUP_SIZE)
                    q_group_offset: u32 = group * packed_per_group

                    # Load all 64 packed bytes of the group (= 128 nibbles)
                    # in one shot and unpack with the wide AIE2P intrinsic.
                    # Saves one q-load + one unpack op + one byte-extract
                    # per group vs the two-chunk variant; the dequant chain
                    # below still runs at 32-lane (no v64accfloat_to_v64bf16).
                    q_packed: aie_vector[u8, 64] = load_v(q_row + q_group_offset, 64)
                    nibbles_all: aie_vector[u8, 128] = unpack_I1024_I8_I4(q_packed, i32(0))
                    # Output layout (from intrinsic_metadata.yaml): nibbles
                    # appear in K-natural order [byte0_lo, byte0_hi, byte1_lo,
                    # byte1_hi, ...], so nibbles_all[i*32:(i+1)*32] covers
                    # K=i*32..i*32+31 of the group.
                    nib_0: aie_vector[u8, 32] = vector_extract(nibbles_all, 0, 32)
                    nib_1: aie_vector[u8, 32] = vector_extract(nibbles_all, 32, 32)
                    nib_2: aie_vector[u8, 32] = vector_extract(nibbles_all, 64, 32)
                    nib_3: aie_vector[u8, 32] = vector_extract(nibbles_all, 96, 32)

                    # === Chunk 0 (K_offset = 0..63 within group) ===
                    # 64-lane Fix2Float front-end (mirrors clang aie::to_float):
                    #   u8 nibbles -> acc32 (ONE ups, unsigned) -> 64-lane i32
                    #   -> +magic (ACC2048.add) -> bitcast accfloat
                    #   -> -magic (ACC2048.accfloat.sub) -> v32->bf16 x2.
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
                    # w_scaled = w_bf * scale (1 bf16 mul per half).
                    w_lo_s_0: aie_vector[bf16, 32] = vector_mul(w_lo_bf_0, scale_v)
                    w_hi_s_0: aie_vector[bf16, 32] = vector_mul(w_hi_bf_0, scale_v)
                    # Concat lo/hi into 64-lane to feed the wide MAC/MSC.
                    w_combined_0: aie_vector[bf16, 64] = concat(w_lo_s_0, w_hi_s_0)
                    # x for K=0..63 of this group is consecutive in memory --
                    # load as a single 64-lane vector instead of 2x 32-lane.
                    x_combined_0: aie_vector[bf16, 64] = load_v(x_in + x_group_offset, 64)
                    acc = I1024_I1024_ACC2048_bf_mac_conf(
                        x_combined_0, w_combined_0, acc, CONF_BF16_MAC
                    )
                    acc = I1024_I1024_ACC2048_bf_msc_conf(
                        x_combined_0, zs_v_64, acc, CONF_BF16_MAC
                    )

                    # === Chunk 1 (K_offset = 64..127 within group) ===
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

        # Single tree reduction on the 64-lane accumulator.
        s: f32 = reduce_add_reassoc(acc)
        p_c[row] = bf16(s)
        row = row + u32(1)

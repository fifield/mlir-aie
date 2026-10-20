// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// One 64x64 tile of C = A * W^T. A is resident in L1 as [kt][row][64];
// W streams into staging as pairs of output columns, bypassing L1. Each
// K-tile's 64 row MMACs overlap the next tile's 32 stream pops.
// gemm_stage_b primes a strip; gemm_tile prefetches across calls using
// persistent staging; gemm_tile_last ends the strip without prefetch.
// C holds CT tiles side by side. OUT_I8 rounds ties upward and saturates
// after OUT_SHIFT; otherwise C holds raw int32 (int8 inputs) or fp32
// (fp8/bf16 inputs) accumulators. Float inputs can instead narrow C to
// OUT_BF16 or OUT_FP8 (E4M3), which cuts DDR write traffic.
//
// DTYPE_FP8 (E4M3), DTYPE_BF16 and DTYPE_I4 reuse the int8 dataflow: a K-tile
// is 64 bytes of A per row and 4096 bytes of W, i.e. 64 fp8, 32 bf16 or 128
// int4 elements. K_DIM counts elements. Integer types also take A_UNS / W_UNS
// to treat A / W as unsigned, and OUT_U8 to narrow C to unsigned int8.

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

using namespace aie;
using namespace me_primitive;

#if defined(DTYPE_FP8) + defined(DTYPE_BF16) + defined(DTYPE_I4) > 1
#error "at most one of DTYPE_FP8, DTYPE_BF16, DTYPE_I4"
#endif
#ifndef A_UNS
#define A_UNS 0
#endif
#ifndef W_UNS
#define W_UNS 0
#endif
#ifndef OUT_U8
#define OUT_U8 0
#endif
#ifndef K_DIM
#define K_DIM 1024
#endif
#ifndef OUT_I8
#define OUT_I8 1
#endif
#ifndef OUT_BF16
#define OUT_BF16 0
#endif
#ifndef OUT_FP8
#define OUT_FP8 0
#endif
#ifndef OUT_SHIFT
#define OUT_SHIFT 12
#endif
#ifndef CT
#define CT 4
#endif
// TIMING overwrites each tile's first four bytes with its completion cycle.
#ifndef TIMING
#define TIMING 0
#endif
// TRACE emits event0/event1 at tile start/end.
#ifndef TRACE
#define TRACE 0
#endif
// A_SPLIT=2 uses two [kt][32][64] halves; 1 uses one [kt][64][64] buffer.
#ifndef A_SPLIT
#define A_SPLIT 2
#endif

constexpr unsigned MR = 64; // A rows per tile = accumulators
#ifdef DTYPE_BF16
constexpr unsigned K_TILE = 32; // K elements per MMAC (32x64 bf16)
#elif defined(DTYPE_I4)
constexpr unsigned K_TILE = 128; // K elements per MMAC (128x64 int4)
#else
constexpr unsigned K_TILE = 64; // K elements per MMAC (64x64 int8/fp8)
#endif
constexpr unsigned KT = K_DIM / K_TILE; // K-tiles per output tile
constexpr unsigned W_STREAM = 1;        // core input stream carrying W
static_assert(K_DIM % K_TILE == 0 && KT >= 2,
              "K must be a multiple of the MMAC K-tile, with >= 2 tiles");
// Restrict conversion to right shifts; larger encodings select left shifts.
static_assert(OUT_SHIFT >= 0 && OUT_SHIFT <= 59,
              "OUT_SHIFT must be in [0, 59]");

#if defined(DTYPE_FP8) || defined(DTYPE_BF16)
static_assert(!OUT_I8, "float inputs produce float output");
#if OUT_BF16
using out_t = bfloat16;
#elif OUT_FP8
using out_t = int8_t; // E4M3 bytes
#else
using out_t = float;
#endif
using acc_t = v64accfloat;
#elif OUT_I8
using out_t = int8_t;
using acc_t = v64acc32;
#elif OUT_U8
using out_t = uint8_t;
using acc_t = v64acc32;
#else
using out_t = int32_t;
using acc_t = v64acc32;
#endif

// One 64-byte A row of a K-tile, W staging load and MMAC for the data type.
#if defined(DTYPE_FP8)
using a_row_t = v64float8;
static inline __attribute__((always_inline)) void pop_w(unsigned idx) {
  insert_staging(get_ss_v128float8_weight(W_STREAM), idx);
}
static inline __attribute__((always_inline)) void load_matrix() {
  staging_to_matrix_m64x64float8();
}
static inline __attribute__((always_inline)) acc_t mac_row(a_row_t a, acc_t acc,
                                                           bool zero) {
  return mac_conf(a, acc, zero);
}
#elif defined(DTYPE_BF16)
using a_row_t = v32bfloat16;
static inline __attribute__((always_inline)) void pop_w(unsigned idx) {
  insert_staging(get_ss_v64bfloat16_weight(W_STREAM), idx);
}
static inline __attribute__((always_inline)) void load_matrix() {
  staging_to_matrix_m32x64bfloat16();
}
static inline __attribute__((always_inline)) acc_t mac_row(a_row_t a, acc_t acc,
                                                           bool zero) {
  return mac_conf(a, acc, zero);
}
#elif defined(DTYPE_I4)
using a_row_t = v128int4;
// Pop one v256int4 (two W columns of one K-tile) into staging slot idx.
static inline __attribute__((always_inline)) void pop_w(unsigned idx) {
  insert_staging(get_ss_v256int4_weight(W_STREAM), idx, MMAC_INT4);
}
static inline __attribute__((always_inline)) void load_matrix() {
  staging_to_matrix_m128x64int4();
}
static inline __attribute__((always_inline)) acc_t mac_row(a_row_t a, acc_t acc,
                                                           bool zero) {
  return mac_conf(a, 1, 1, acc, zero);
}
#else
#if A_UNS
using a_row_t = v64uint8;
#else
using a_row_t = v64int8;
#endif
// Pop one v128int8 (two W columns of one K-tile) into staging slot idx.
static inline __attribute__((always_inline)) void pop_w(unsigned idx) {
#if W_UNS
  insert_staging(get_ss_v128uint8_weight(W_STREAM), idx, MMAC_UINT8);
#else
  insert_staging(get_ss_v128int8_weight(W_STREAM), idx, MMAC_INT8);
#endif
}
static inline __attribute__((always_inline)) void load_matrix() {
  staging_to_matrix_m64x64int8();
}
static inline __attribute__((always_inline)) acc_t mac_row(a_row_t a, acc_t acc,
                                                           bool zero) {
  return mac_conf(a, !A_UNS, !W_UNS, acc, zero);
}
#endif

template <bool PREFETCH>
static inline __attribute__((always_inline)) void
gemm_tile_impl(const int8_t *restrict A_lo, const int8_t *restrict A_hi,
               out_t *restrict C, int j) {
#if TRACE
  event0();
#endif
  // Rows 0-31 from pLo, rows 32-63 from pHi; both advance by kt_step rows.
  constexpr unsigned kt_step = A_SPLIT == 2 ? MR / 2 : MR;
  const a_row_t *pLo = (const a_row_t *)A_lo;
  const a_row_t *pHi =
      A_SPLIT == 2 ? (const a_row_t *)A_hi : (const a_row_t *)A_lo + MR / 2;
  acc_t acc[MR];
  bool zero = 1;

  for (unsigned kt = 0; kt < KT - 1; kt++)
    chess_prepare_for_pipelining chess_loop_range(KT - 1, KT - 1) {
      load_matrix();
#pragma unroll
      for (unsigned r = 0; r < MR; r++) {
        acc[r] = mac_row(r < MR / 2 ? pLo[r] : pHi[r - MR / 2], acc[r], zero);
        if (r & 1)
          pop_w(r / 2);
      }
      zero = 0;
      pLo += kt_step;
      pHi += kt_step;
    }

  // Last K-tile: optionally pre-stage the next output tile's first K-tile.
  load_matrix();
#pragma unroll
  for (unsigned r = 0; r < MR; r++) {
    acc[r] = mac_row(r < MR / 2 ? pLo[r] : pHi[r - MR / 2], acc[r], 0);
    if constexpr (PREFETCH)
      if (r & 1)
        pop_w(r / 2);
  }

#if OUT_I8
  v64int8 *pC = (v64int8 *)(C + j * 64);
#pragma unroll
  for (unsigned r = 0; r < MR; r++)
    pC[r * CT] =
        aie::accum<acc32, 64>(acc[r]).template to_vector<int8>(OUT_SHIFT);
#elif OUT_U8
  v64uint8 *pC = (v64uint8 *)(C + j * 64);
#pragma unroll
  for (unsigned r = 0; r < MR; r++)
    pC[r * CT] =
        aie::accum<acc32, 64>(acc[r]).template to_vector<uint8>(OUT_SHIFT);
#elif OUT_BF16
  aie::vector<bfloat16, 64> *pC = (aie::vector<bfloat16, 64> *)(C + j * 64);
#pragma unroll
  for (unsigned r = 0; r < MR; r++)
    pC[r * CT] =
        aie::accum<accfloat, 64>(acc[r]).template to_vector<bfloat16>();
#elif OUT_FP8
  aie::vector<float8, 64> *pC = (aie::vector<float8, 64> *)(C + j * 64);
#pragma unroll
  for (unsigned r = 0; r < MR; r++)
    pC[r * CT] = aie::accum<accfloat, 64>(acc[r]).template to_vector<float8>();
#else
  acc_t *pC = (acc_t *)(C + j * 64);
#pragma unroll
  for (unsigned r = 0; r < MR; r++)
    pC[r * CT] = acc[r];
#endif
#if TIMING
  *(volatile int32_t *)(C + j * 64) = (int32_t)get_cycles();
#endif
#if TRACE
  event1();
#endif
}

extern "C" {

void gemm_stage_b() {
#if OUT_I8 || OUT_U8
  aie::set_rounding(aie::rounding_mode::positive_inf);
  aie::set_saturation(aie::saturation_mode::saturate);
#elif OUT_BF16 || OUT_FP8
  aie::set_rounding(aie::rounding_mode::conv_even);
  aie::set_saturation(aie::saturation_mode::saturate);
#endif
#pragma unroll
  for (unsigned i = 0; i < 32; i++)
    pop_w(i);
}

void gemm_tile(const int8_t *restrict A_lo, const int8_t *restrict A_hi,
               out_t *restrict C, int j) {
  gemm_tile_impl<true>(A_lo, A_hi, C, j);
}

void gemm_tile_last(const int8_t *restrict A_lo, const int8_t *restrict A_hi,
                    out_t *restrict C, int j) {
  gemm_tile_impl<false>(A_lo, A_hi, C, j);
}

} // extern "C"

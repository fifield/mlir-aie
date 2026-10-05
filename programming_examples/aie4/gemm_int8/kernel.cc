// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// One 64x64 tile of C = A * W^T. A is resident in L1 as [kt][row][64];
// W streams into staging as pairs of output columns, bypassing L1. Each
// K-tile's 64 row MMACs overlap the next tile's 32 stream pops.
// gemm_stage_b primes a strip; gemm_tile prefetches across calls using
// persistent staging; gemm_tile_last ends the strip without prefetch.
// C holds CT tiles side by side. OUT_I8 rounds ties upward and saturates
// after OUT_SHIFT; otherwise C holds raw int32 accumulators.

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

using namespace aie;
using namespace me_primitive;

#ifndef K_DIM
#define K_DIM 1024
#endif
#ifndef OUT_I8
#define OUT_I8 1
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

constexpr unsigned MR = 64;         // A rows per tile = accumulators
constexpr unsigned KT = K_DIM / 64; // K-tiles per output tile
constexpr unsigned W_STREAM = 1;    // core input stream carrying W
static_assert(K_DIM % 64 == 0 && KT >= 2, "K must be a multiple of 64, >= 128");
// Restrict conversion to right shifts; larger encodings select left shifts.
static_assert(OUT_SHIFT >= 0 && OUT_SHIFT <= 59,
              "OUT_SHIFT must be in [0, 59]");

#if OUT_I8
using out_t = int8_t;
#else
using out_t = int32_t;
#endif

// Pop one v128int8 (two W columns of one K-tile) into staging slot idx.
static inline __attribute__((always_inline)) void pop_w(unsigned idx) {
  insert_staging(get_ss_v128int8_weight(W_STREAM), idx, MMAC_INT8);
}

template <bool PREFETCH>
static inline __attribute__((always_inline)) void
gemm_tile_impl(const int8_t *restrict A_lo, const int8_t *restrict A_hi,
               out_t *restrict C, int j) {
#if TRACE
  event0();
#endif
  // Rows 0-31 from pLo, rows 32-63 from pHi; both advance by kt_step rows.
  constexpr unsigned kt_step = A_SPLIT == 2 ? MR / 2 : MR;
  const v64int8 *pLo = (const v64int8 *)A_lo;
  const v64int8 *pHi =
      A_SPLIT == 2 ? (const v64int8 *)A_hi : (const v64int8 *)A_lo + MR / 2;
  v64acc32 acc[MR];
  bool zero = 1;

  for (unsigned kt = 0; kt < KT - 1; kt++)
    chess_prepare_for_pipelining chess_loop_range(KT - 1, KT - 1) {
      staging_to_matrix_m64x64int8();
#pragma unroll
      for (unsigned r = 0; r < MR; r++) {
        acc[r] =
            mac_conf(r < MR / 2 ? pLo[r] : pHi[r - MR / 2], 1, 1, acc[r], zero);
        if (r & 1)
          pop_w(r / 2);
      }
      zero = 0;
      pLo += kt_step;
      pHi += kt_step;
    }

  // Last K-tile: optionally pre-stage the next output tile's first K-tile.
  staging_to_matrix_m64x64int8();
#pragma unroll
  for (unsigned r = 0; r < MR; r++) {
    acc[r] = mac_conf(r < MR / 2 ? pLo[r] : pHi[r - MR / 2], 1, 1, acc[r], 0);
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
#else
  v64acc32 *pC = (v64acc32 *)(C + j * 64);
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
#if OUT_I8
  aie::set_rounding(aie::rounding_mode::positive_inf);
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

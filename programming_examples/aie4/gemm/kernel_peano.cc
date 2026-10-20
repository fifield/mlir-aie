// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// One 64x64 tile of C = A * W^T. A is resident in L1 as [kt][row][64];
// W streams into staging as pairs of output columns, bypassing L1. Each
// K-tile's 64 row MMACs overlap the next tile's 32 stream pops.
// gemm_stage_b primes a strip; gemm_tile prefetches across calls using
// persistent staging; gemm_tile_last ends the strip without prefetch.
// C holds CT tiles side by side. OUT_I8 rounds ties upward and saturates
// after OUT_SHIFT; otherwise C holds raw int32 accumulators.

// Peano (LLVM) port of kernel.cc: same entry points, macros and semantics, but
// written with Peano builtins (no aie_api). Build with:
//   clang++ --target=aie4-none-unknown-elf -O2 -std=c++20 \
//     -mllvm -aie4postlegalizercustomcombiner-disable-rule=combine_staging_upd_load
// The full-unroll pragma must be `#pragma clang loop unroll(full)`; plain
// `#pragma unroll` makes Peano spill the 64 accumulators to the stack.
// aie4intrin.h is included implicitly by the aie4 target.
#include <stdint.h>

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
// TIMING (unsupported here, see below) would overwrite each tile's first four
// bytes with its completion cycle.
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

#if TIMING
// Peano has no usable cycle counter: get_cycles() is declared but never defined
// (chess reads the CNTR register), and read_tm() of core-module registers
// (Timer_Low, Event_Status0, Core_Status) hangs the core on npu3. Use TRACE.
#error "TIMING is not supported by the Peano kernel; use TRACE_SIZE instead"
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

  for (unsigned kt = 0; kt < KT - 1; kt++) {
      staging_to_matrix_m64x64int8();
#pragma clang loop unroll(full)
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
#pragma clang loop unroll(full)
  for (unsigned r = 0; r < MR; r++) {
    acc[r] = mac_conf(r < MR / 2 ? pLo[r] : pHi[r - MR / 2], 1, 1, acc[r], 0);
    if constexpr (PREFETCH)
      if (r & 1)
        pop_w(r / 2);
  }

#if OUT_I8
  v64int8 *pC = (v64int8 *)(C + j * 64);
#pragma clang loop unroll(full)
  for (unsigned r = 0; r < MR; r++)
    pC[r * CT] = to_v64int8(acc[r], OUT_SHIFT, 1);
#else
  v64acc32 *pC = (v64acc32 *)(C + j * 64);
#pragma clang loop unroll(full)
  for (unsigned r = 0; r < MR; r++)
    pC[r * CT] = acc[r];
#endif
#if TRACE
  event1();
#endif
}

extern "C" {

void gemm_stage_b() {
#if OUT_I8
  set_rnd(rnd_pos_inf);
  set_satmode(1); // saturate
#endif
#pragma clang loop unroll(full)
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

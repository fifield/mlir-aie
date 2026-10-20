// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Time one 64x64 MMAC tile, with A in L1 and B streamed into staging.
// Prefetched B persists in staging across calls; the final call skips prefetch.
// Peano port of kernel.cc (see ../kernel_peano.cc). aie4intrin.h is implicit.
#include <stdint.h>
// Peano declares get_cycles() but never defines it (the chess version reads the
// CNTR register), and read_tm() of core-module registers (Timer_Low 0xA0120,
// Event_Status0, Core_Status) hangs the core on npu3. So the Peano build has no
// cycle counter: the kernel is checked for correctness and the cycle columns
// print 0. Use the static bundle count and the GEMM trace (TRACE_SIZE) instead.
static inline __attribute__((always_inline)) uint32_t peano_cycles() {
  return 0;
}

#ifndef KT
#define KT 8
#endif
constexpr unsigned MR = 64;
constexpr unsigned B_STREAM = 1;

static inline __attribute__((always_inline)) void pop_b(unsigned idx,
                                                        unsigned mode) {
  insert_staging(get_ss_v128int8_weight(B_STREAM), idx, mode);
}

template <bool PREFETCH>
static inline __attribute__((always_inline)) void
mm_tile(const int8_t *restrict A, int32_t *restrict C) {
  uint32_t t0 = peano_cycles();
  const v64int8 *pA = (const v64int8 *)A;
  v64acc32 acc[MR];
  unsigned mode = MMAC_INT8;
  bool zero = 1;
  for (unsigned kt = 0; kt < KT - 1; kt++) {
      staging_to_matrix_m64x64int8();
#pragma clang loop unroll(full)
      for (unsigned r = 0; r < MR; r++) {
        acc[r] = mac_conf(pA[r], 1, 1, acc[r], zero);
        if (r & 1)
          pop_b(r / 2, mode);
      }
      zero = 0;
      pA += MR;
    }
  // Last K-tile: optionally prefetch the next output tile's first B tile.
  staging_to_matrix_m64x64int8();
#pragma clang loop unroll(full)
  for (unsigned r = 0; r < MR; r++) {
    acc[r] = mac_conf(pA[r], 1, 1, acc[r], 0);
    if constexpr (PREFETCH)
      if (r & 1)
        pop_b(r / 2, mode);
  }
  uint32_t t1 = peano_cycles();
  v64acc32 *pC = (v64acc32 *)C;
#pragma clang loop unroll(full)
  for (unsigned r = 0; r < MR; r++)
    pC[r] = acc[r];
  uint32_t t2 = peano_cycles();
  C[MR * 64 + 0] = (int32_t)(t1 - t0);
  C[MR * 64 + 1] = (int32_t)(t2 - t0);
}

extern "C" {
void gemm_stage_b() {
  unsigned mode = MMAC_INT8;
#pragma clang loop unroll(full)
  for (unsigned i = 0; i < 32; i++)
    pop_b(i, mode);
}
void gemm_tile(const int8_t *restrict A, int32_t *restrict C) {
  mm_tile<true>(A, C);
}
void gemm_tile_last(const int8_t *restrict A, int32_t *restrict C) {
  mm_tile<false>(A, C);
}
}

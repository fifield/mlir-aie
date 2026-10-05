// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Time one 64x64 MMAC tile, with A in L1 and B streamed into staging.
// Prefetched B persists in staging across calls; the final call skips prefetch.
#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>
using namespace aie;
using namespace me_primitive;

#ifndef KT
#define KT 8
#endif
constexpr unsigned MR = 64;
constexpr unsigned B_STREAM = 1;

static inline __attribute__((always_inline)) void pop_b(unsigned idx,
                                                        uint2_t mode) {
  insert_staging(get_ss_v128int8_weight(B_STREAM), idx, mode);
}

template <bool PREFETCH>
static inline __attribute__((always_inline)) void
mm_tile(const int8_t *restrict A, int32_t *restrict C) {
  uint64_t t0 = get_cycles();
  const v64int8 *pA = (const v64int8 *)A;
  v64acc32 acc[MR];
  uint2_t mode = MMAC_INT8;
  bool zero = 1;
  for (unsigned kt = 0; kt < KT - 1; kt++)
    chess_prepare_for_pipelining chess_loop_range(KT - 1, KT - 1) {
      staging_to_matrix_m64x64int8();
#pragma unroll
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
#pragma unroll
  for (unsigned r = 0; r < MR; r++) {
    acc[r] = mac_conf(pA[r], 1, 1, acc[r], 0);
    if constexpr (PREFETCH)
      if (r & 1)
        pop_b(r / 2, mode);
  }
  uint64_t t1 = get_cycles();
  v64acc32 *pC = (v64acc32 *)C;
#pragma unroll
  for (unsigned r = 0; r < MR; r++)
    pC[r] = acc[r];
  uint64_t t2 = get_cycles();
  C[MR * 64 + 0] = (int32_t)(t1 - t0);
  C[MR * 64 + 1] = (int32_t)(t2 - t0);
}

extern "C" {
void gemm_stage_b() {
  uint2_t mode = MMAC_INT8;
#pragma unroll
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

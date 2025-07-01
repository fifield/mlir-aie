//===- kernel.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>
#include <stdio.h>

using namespace aie;

inline __attribute__((always_inline)) void matrix_staged_load(v128int8 *q,
                                                              int mode)
    __attribute__((overloadable)) {

  v128int8 *pB = q;
  #pragma unroll
  for (size_t i = 0; i < 32; i++) {
    insert_staging(*pB, i, mode);
    pB++;
  }
}

extern "C" {
void matmul(int8_t *restrict inIterA, int8_t *restrict inIterB,
            int32_t *restrict outIterC);
}

void matmul(int8_t *restrict inIterA, int8_t *restrict inIterB,
            int32_t *restrict outIterC) {

  constexpr unsigned int K_outer = 4;
  constexpr unsigned int M = 4;

  v64acc32 acc[M];
  #pragma unroll
  for (unsigned l = 0; l < M; l++) {
    acc[l] = undef_v64acc32();
  }

  uint2_t mul_mode = MMAC_INT8;
  bool zero_acc = 1; // Accumulation set to zero in first iteration

  for (unsigned int k = 0; k < K_outer; k++)
    chess_prepare_for_pipelining {

      unsigned int B_offset = k * 32; // reading 32 v128int8 values
      v128int8 *B_mat_ptr =
          ((v128int8 *)inIterB) + B_offset; // B is matrix64x64

      matrix_staged_load(B_mat_ptr, mul_mode);
      staging_to_matrix_m64x64int8();

      #pragma unroll
      for (unsigned int i = 0; i < M; i++) {
        unsigned int offset = (k * M + i) * 64;
        v64int8 input_vector = *((v64int8 *)((int8_t *)inIterA + offset));

        acc[i] = mac_conf(input_vector, 1, 1, acc[i], zero_acc);
      }

      zero_acc = 0; // Accumulation mode set after first iteration
    }

  v64acc32 *restrict pOut = (v64acc32 *)outIterC;

  #pragma unroll
  for (unsigned l = 0; l < M; l++) {
    *pOut++ = acc[l];
  }
}

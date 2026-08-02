//===- kernel.cc - MMAC int16*int8 matrix multiply -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Mixed-precision matrix multiply C[Mx64] += A[MxK] * B[Kx64]
// with int16 A inputs and int8 B inputs, int32 output.
// M=4 rows, K=128 (2 outer iterations of 64), N=64 columns.
// Uses AIE4 MMAC unit via emulated mode (ISA Table 195): cast int16 A
// to int8 (truncate to lower 8 bits) and use 64x64 int8 matrix register.
// For true int16*int8 without truncation, the aie::mmul API would be needed.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

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
void matmul(int16_t *restrict A, int8_t *restrict B, int32_t *restrict C);
}

void matmul(int16_t *restrict A, int8_t *restrict B, int32_t *restrict C) {
  constexpr unsigned int K_outer = 2;
  constexpr unsigned int M = 4;

  v64acc32 acc[M];
#pragma unroll
  for (unsigned l = 0; l < M; l++)
    acc[l] = undef_v64acc32();

  uint2_t mul_mode = MMAC_INT8;
  bool zero_acc = 1;

  for (unsigned int k = 0; k < K_outer; k++)
    chess_prepare_for_pipelining {
      unsigned int B_offset = k * 32;
      v128int8 *B_mat_ptr = ((v128int8 *)B) + B_offset;

      matrix_staged_load(B_mat_ptr, mul_mode);
      staging_to_matrix_m64x64int8();

#pragma unroll
      for (unsigned int i = 0; i < M; i++) {
        unsigned int offset = (k * M + i) * 64;
        // Cast int16 A to int8 via pack — truncates to lower 8 bits
        v64int16 a16 = *((v64int16 *)((int16_t *)A + offset));
        v64int8 input_vector = pack(a16);
        acc[i] = mac_conf(input_vector, 1, 1, acc[i], zero_acc);
      }

      zero_acc = 0;
    }

  v64acc32 *restrict pOut = (v64acc32 *)C;
#pragma unroll
  for (unsigned l = 0; l < M; l++)
    *pOut++ = acc[l];
}

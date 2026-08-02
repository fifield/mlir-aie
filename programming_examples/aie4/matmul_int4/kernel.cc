//===- kernel.cc - MMAC int4*int4 matrix multiply ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Matrix multiply C[Mx64] += A[MxK] * B[Kx64] with int4 packed inputs.
// M=4 rows, K=512 (4 outer iterations of 128), N=64 columns.
// Uses AIE4 MMAC unit: 128x64 int4 matrix register, 8192 mults/cycle.
// ISA: MMAC.i int4*int4 (bmode=0).
// Staging: load_staging(v256int4*) + staging_to_matrix_m128x64int4().
// A input: v256int4 (2 rows of 128 int4), extract_v128int4 per row.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

using namespace aie;

extern "C" {
void matmul(int8_t *restrict A, int8_t *restrict B, int32_t *restrict C);
}

void matmul(int8_t *restrict A, int8_t *restrict B, int32_t *restrict C) {
  constexpr unsigned int K_outer = 4;
  constexpr unsigned int M = 4;

  v64acc32 acc[M];
#pragma unroll
  for (unsigned l = 0; l < M; l++)
    acc[l] = undef_v64acc32();

  bool zero_acc = 1;

  for (unsigned int k = 0; k < K_outer; k++)
    chess_prepare_for_pipelining {
      v256int4 *pB = (v256int4 *)((int8_t *)B + k * 4096);
      load_staging(pB);
      staging_to_matrix_m128x64int4();

#pragma unroll
      for (unsigned int i = 0; i < M / 2; i++) {
        unsigned int offset = (k * M / 2 + i) * 128;
        v256int4 Xbuff = *((v256int4 *)((int8_t *)A + offset));
        acc[2 * i] =
            mac_conf(extract_v128int4(Xbuff, 0), 1, 1, acc[2 * i], zero_acc);
        acc[2 * i + 1] =
            mac_conf(extract_v128int4(Xbuff, 1), 1, 1, acc[2 * i + 1], zero_acc);
      }

      zero_acc = 0;
    }

  v64acc32 *restrict pOut = (v64acc32 *)C;
#pragma unroll
  for (unsigned l = 0; l < M / 2; l++) {
    *pOut++ = acc[2 * l];
    *pOut++ = acc[2 * l + 1];
  }
}

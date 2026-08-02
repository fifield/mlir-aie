//===- kernel.cc - MMAC fp8*fp8 matrix multiply --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Matrix multiply C[Mx64] += A[MxK] * B[Kx64] with fp8 (E4M3) inputs,
// fp32 output. M=4 rows, K=256 (4 outer iterations of 64), N=64 columns.
// Uses AIE4 MMAC unit: 64x64 fp8 matrix register, 4096 mults/cycle.
// ISA: MMAC.f fp8*fp8 (14.5).
// Staging: load_staging(v128float8*) + staging_to_matrix_m64x64float8().
// A input: v128float8 (2 rows of 64 fp8), extract_v64float8 per row.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

using namespace aie;

extern "C" {
void matmul(int8_t *restrict A, int8_t *restrict B, float *restrict C);
}

void matmul(int8_t *restrict A, int8_t *restrict B, float *restrict C) {
  constexpr unsigned int K_outer = 4;
  constexpr unsigned int M = 4;

  v64accfloat acc[M];
#pragma unroll
  for (unsigned l = 0; l < M; l++)
    acc[l] = undef_v64accfloat();

  bool zero_acc = 1;

  for (unsigned int k = 0; k < K_outer; k++)
    chess_prepare_for_pipelining {
      v128float8 *pB = (v128float8 *)((int8_t *)B + k * 4096);
      load_staging(pB);
      staging_to_matrix_m64x64float8();

#pragma unroll
      for (unsigned int i = 0; i < M / 2; i++) {
        unsigned int offset = (k * M / 2 + i) * 128;
        v128float8 Xbuff = *((v128float8 *)((int8_t *)A + offset));
        acc[2 * i] =
            mac_conf(extract_v64float8(Xbuff, 0), acc[2 * i], zero_acc);
        acc[2 * i + 1] =
            mac_conf(extract_v64float8(Xbuff, 1), acc[2 * i + 1], zero_acc);
      }

      zero_acc = 0;
    }

  v64accfloat *restrict pOut = (v64accfloat *)C;
#pragma unroll
  for (unsigned l = 0; l < M / 2; l++) {
    *pOut++ = acc[2 * l];
    *pOut++ = acc[2 * l + 1];
  }
}

//===- kernel.cc - MMAC bf16*bf16 matrix multiply ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Matrix multiply C[Mx64] += A[MxK] * B[Kx64] with bf16 inputs, fp32 output.
// M=4 rows, K=128 (4 outer iterations of 32), N=64 columns.
// Uses AIE4 MMAC unit: 32x64 bf16 matrix register, 2048 mults/cycle.
// ISA: MMAC.f bf16*bf16 on AIE4b.
// Staging: load_staging(v64bfloat16*) + staging_to_matrix_m32x64bfloat16().
// A input: v64bfloat16 (2 rows of 32 bf16), extract_v32bfloat16 per row.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

using namespace aie;

extern "C" {
void matmul(bfloat16 *restrict A, bfloat16 *restrict B, float *restrict C);
}

void matmul(bfloat16 *restrict A, bfloat16 *restrict B, float *restrict C) {
  constexpr unsigned int K_outer = 4;
  constexpr unsigned int M = 4;

  v64accfloat acc[M];
#pragma unroll
  for (unsigned l = 0; l < M; l++)
    acc[l] = undef_v64accfloat();

  bool zero_acc = 1;

  for (unsigned int k = 0; k < K_outer; k++)
    chess_prepare_for_pipelining {
      v64bfloat16 *pB = (v64bfloat16 *)((bfloat16 *)B + k * 2048);
      load_staging(pB);
      staging_to_matrix_m32x64bfloat16();

#pragma unroll
      for (unsigned int i = 0; i < M / 2; i++) {
        unsigned int offset = (k * M / 2 + i) * 64;
        v64bfloat16 Xbuff = *((v64bfloat16 *)((bfloat16 *)A + offset));
        acc[2 * i] =
            mac_conf(extract_v32bfloat16(Xbuff, 0), acc[2 * i], zero_acc);
        acc[2 * i + 1] =
            mac_conf(extract_v32bfloat16(Xbuff, 1), acc[2 * i + 1], zero_acc);
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

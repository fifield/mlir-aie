//===- kernel.cc - MMAC MXINT8 matrix multiply ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Matrix multiply C[Mx64] += A[MxK] * B[Kx64] with MXINT8 packed inputs.
// MXINT8: S8 mantissa (6 frac bits) + E8M0 shared exponent, block_size=32 (ebs32).
// M=4 rows, K=256 (4 outer iterations of 64), N=64 columns, fp32 output.
// Uses AIE4 MMAC unit with FIFO-based MX staging (mxint8 types).
// Per 128-element vector: 4 blocks of [1 SF + 32 mantissa] = 132 bytes.
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

  v128mxint8_unaligned *restrict pB = (v128mxint8_unaligned *)B;
  v128mxint8_unaligned *restrict pA = (v128mxint8_unaligned *)A;

  unsigned mxVecBytes = 132; // 128 int8 mantissa + 4 E8M0 exponents (ebs32)

  fifo_state_t fB, fA;
  sparse_fifo_t chess_storage(lf0) fAf;
  sparse_fifo_t chess_storage(lf1) fBf;
  fA.fifo = fAf;
  fB.fifo = fBf;

  bool zero_acc = 1;

  for (unsigned int k = 0; k < K_outer; k++)
    chess_prepare_for_pipelining {
      unsigned int B_offset = k * 32 * mxVecBytes;
      pB = (v128mxint8_unaligned *)((int8_t *)B + B_offset);

      fB.fifo = chess_dont_care(sparse_fifo_t);
      sparse_fifo_t chess_storage(lf1) tmpB = chess_copy(fB.fifo);
      fB.fifo = tmpB;
      fifo_ld_reset(pB, fB);

      unsigned int idx = 0;
#pragma unroll
      for (unsigned int j = 0; j < 32; j++)
        fifo_ld_pop_staging(pB, fB, idx++);

      staging_to_matrix_m64x64mxint8();

      unsigned int A_offset = k * M / 2 * mxVecBytes;
      pA = (v128mxint8_unaligned *)((int8_t *)A + A_offset);

      fA.fifo = chess_dont_care(sparse_fifo_t);
      sparse_fifo_t chess_storage(lf0) tmpA = chess_copy(fA.fifo);
      fA.fifo = tmpA;
      fifo_ld_reset(pA, fA);

#pragma unroll
      for (unsigned int i = 0; i < M / 2; i++) {
        v128mxint8 a = fifo_ld_pop(pA, fA);
        v64mxint8 Xbuff = extract_v64mxint8(a, 0);
        v64mxint8 Xbuff2 = extract_v64mxint8(a, 1);

        acc[i] = mac_conf(Xbuff, acc[i], zero_acc);
        acc[i + M / 2] = mac_conf(Xbuff2, acc[i + M / 2], zero_acc);
      }

      zero_acc = 0;
    }

  v64accfloat *restrict pOut = (v64accfloat *)C;
#pragma unroll
  for (unsigned l = 0; l < M / 2; l++) {
    *pOut++ = acc[l];
    *pOut++ = acc[l + M / 2];
  }
}

//===- kernel.cc - MMAC fp8*fp8 sparse matrix multiply -------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Sparse matrix multiply C[Mx64] += A[Mx128] * B_sparse[128x64] with fp8
// inputs and fp32 output, at 4:2 structured sparsity.
// M=4 rows, N=64 columns.
// Uses the AIE4 MMAC sparse unit: 128x64 fp8 sparse matrix register (QF_Zb).
// ISA: MMAC.f sparse.  xsrc = Y (128 dense fp8), qf_zsrc = QF_Zb.
//
// Operand shapes:
//   A[4 * 128]   = 512 bytes fp8   (DENSE: 128 logical K per row)
//   B[64 * 64]   = 4096 bytes fp8  (STORED: 2 kept elements per K quartet,
//                                   column-major, 64 bytes per column)
//   mask[192]    = 768 bytes int32 (6-Kbit sparsity register image)
//   C[4 * 64]    = 1024 bytes fp32
//
// Sparsity structure (according to the AIE4 ISA):
// each of the 64 matrix columns is divided into 32 quartets of 4 logical K
// positions, of which exactly 2 are stored.  A per-column, per-quartet 3-bit
// code selects which two, INDEPENDENTLY FOR EVERY COLUMN:
//   001 -> x[0],x[2]   010 -> x[0],x[3]   011 -> x[1],x[2]
//   100 -> x[1],x[3]   101 -> x[2],x[3]   else -> x[0],x[1]
// So the sparsity register holds 64 columns x 32 quartets x 3 bits = 6 Kbit,
// addressed as six 1-Kbit row slices of 16 bits per column.
//
// Flow: load_staging(B, mask) + staging_to_matrix_m128x64float8_sparse(),
//       then mac_conf(v128float8 A_row, ...) per M row.
//
// This is the fp8 analogue of matmul_int8_sparse.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

using namespace aie;

extern "C" {
void matmul(int8_t *restrict A, int8_t *restrict B, float *restrict C,
            uint32_t *restrict mask);
}

void matmul(int8_t *restrict A, int8_t *restrict B, float *restrict C,
            uint32_t *restrict mask) {
  constexpr unsigned int M = 4;

  v64accfloat acc[M];
#pragma unroll
  for (unsigned l = 0; l < M; l++)
    acc[l] = undef_v64accfloat();

  // Stage B (64 stored rows x 64 columns of fp8) together with the full
  // 6-Kbit sparsity register image.  The two-argument load_staging writes
  // all six 1-Kbit row slices of Q; insert_staging_sparsity() writes only
  // one of them and would leave the other five stale.
  v128float8 *pB = (v128float8 *)B;
  load_staging(pB, (v32uint32 *)mask);

  // Copy data + mask to the QF_Zb matrix register
  staging_to_matrix_m128x64float8_sparse();

  // MAC: each A row supplies its 128 dense fp8 activations
#pragma unroll
  for (unsigned int i = 0; i < M; i++) {
    v128float8 Xbuff = *((v128float8 *)((int8_t *)A + i * 128));
    acc[i] = mac_conf(Xbuff, acc[i], 1);
  }

  v64accfloat *restrict pOut = (v64accfloat *)C;
#pragma unroll
  for (unsigned l = 0; l < M; l++)
    *pOut++ = acc[l];
}

//===----------------------------------------------------------------------===//
// Hardware (npu3) entry point.
//
// An AIE4 compute tile has only two input DMA channels, but this kernel
// needs three inputs (A, B and the sparsity mask).  The hardware design
// therefore streams A and the mask as a single contiguous buffer -- A's 512
// bytes followed by the mask's 768 bytes -- and this thin wrapper splits it
// again.
//===----------------------------------------------------------------------===//

extern "C" {
void matmul_hw(int8_t *restrict AM, int8_t *restrict B, float *restrict C);
}

void matmul_hw(int8_t *restrict AM, int8_t *restrict B, float *restrict C) {
  matmul(AM, B, C, (uint32_t *)(AM + 512));
}

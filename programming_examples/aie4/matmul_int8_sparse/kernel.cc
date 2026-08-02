//===- kernel.cc - MMAC int8*int8 sparse matrix multiply -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Sparse matrix multiply C[Mx64] += A[Mx128] * B_sparse[128x64] with int8
// inputs and int32 output, at 4:2 structured sparsity.
//
// Uses the AIE4 MMAC sparse unit: 128x64 int8 sparse matrix register.
// ISA: MMAC.i sparse.  xsrc = 128 dense int8 per A row.
//
// Flow: load_staging(B) + insert_staging_sparsity(&mask)
//       + staging_to_matrix_m128x64int8_sparse()
//
// Operand shapes:
//   A[4 * 128]   = 512 bytes int8  (DENSE: 128 logical K per row)
//   B[64 * 64]   = 4096 bytes int8 (STORED: 2 kept elements per K quartet,
//                                   column-major, 64 bytes per column)
//   mask[192]    = 768 bytes int32 (6-Kbit sparsity register image)
//   C[4 * 64]    = 1024 bytes int32
//
// Sparsity structure (AIE4 ISA Spec v1.1 sections 2.7.2.5 and 14.1.5):
// each of the 64 matrix columns is divided into 32 quartets of 4 logical K
// positions, of which exactly 2 are stored.  A per-column, per-quartet 3-bit
// code selects which two, INDEPENDENTLY FOR EVERY COLUMN:
//   001 -> x[0],x[2]   010 -> x[0],x[3]   011 -> x[1],x[2]
//   100 -> x[1],x[3]   101 -> x[2],x[3]   else -> x[0],x[1]
// So the sparsity register holds 64 columns x 32 quartets x 3 bits = 6 Kbit,
// addressed as six 1-Kbit row slices of 16 bits per column.  Note that a
// quartet's 3 bits may straddle two row slices.
//
// Note: this is the int8 analogue of matmul_fp8_sparse.  AIE4 drives its
// sparse MMAC through the staging path above; the AIE2-style
// v256int8_sparse / mul_4x16_16x16 quartet API does not exist on AIE4.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

using namespace aie;

extern "C" {
void matmul(int8_t *restrict A, int8_t *restrict B, int32_t *restrict C,
            uint32_t *restrict mask);
}

void matmul(int8_t *restrict A, int8_t *restrict B, int32_t *restrict C,
            uint32_t *restrict mask) {
  constexpr unsigned int M = 4;

  v64acc32 acc[M];
#pragma unroll
  for (unsigned l = 0; l < M; l++)
    acc[l] = undef_v64acc32();

  // Stage B (64 stored rows x 64 columns of int8) together with the full
  // 6-Kbit sparsity register image.  The two-argument load_staging writes
  // all six 1-Kbit row slices of Q; insert_staging_sparsity() writes only
  // one of them and would leave the other five stale.
  v128int8 *pB = (v128int8 *)B;
  load_staging(pB, (v32uint32 *)mask);

  // Copy data + mask into the sparse matrix register.
  staging_to_matrix_m128x64int8_sparse();

  // MAC: each A row supplies its 128 dense int8 activations.
#pragma unroll
  for (unsigned int i = 0; i < M; i++) {
    v128int8 Xbuff = *((v128int8 *)((int8_t *)A + i * 128));
    acc[i] = mac_conf(Xbuff, 1, 1, acc[i], /*zero_acc=*/1);
  }

  v64acc32 *restrict pOut = (v64acc32 *)C;
#pragma unroll
  for (unsigned l = 0; l < M; l++)
    *pOut++ = acc[l];
}

//===----------------------------------------------------------------------===//
// Hardware (npu3) entry point.
//
// An AIE4 compute tile has only two input DMA channels, but this kernel
// needs three inputs (A, B and the sparsity mask).  The hardware design
// therefore streams A and the mask as a single contiguous buffer, with the
// 768-byte mask appended after A's 512 bytes, and this thin wrapper splits
// it again.
//===----------------------------------------------------------------------===//

extern "C" {
void matmul_hw(int8_t *restrict AM, int8_t *restrict B, int32_t *restrict C);
}

void matmul_hw(int8_t *restrict AM, int8_t *restrict B, int32_t *restrict C) {
  matmul(AM, B, C, (uint32_t *)(AM + 512));
}

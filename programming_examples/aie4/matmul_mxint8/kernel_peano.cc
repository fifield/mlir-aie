//===- kernel_peano.cc - MMAC MXINT8 matrix multiply (Peano) -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Peano (LLVM) port of kernel.cc: same entry point and semantics, written with
// the Peano AIE4 intrinsics (aie4intrin.h, included implicitly by the aie4
// target) instead of aie_api and chess register-binding hints. A separate file
// is needed because:
//  - Peano has no v128mxint8_unaligned alias (v128mxint8_ebs32_unaligned).
//  - Peano does not tie the pointer/fifo/avail operands of the plain MXINT8
//    pops (VLDB.POP mxi_za[i] and VLDA.POP mxi_y; missing from
//    AIE4InstrInfo::getTiedRegInfo), so register allocation can split one
//    fifo stream across p0/lf0/r24 and p1/lf1/r25 and the MMAC reads garbage.
//    The 1D/2D (jump-mode) pops are tied, so each vector is loaded with its
//    own fill + jump-mode pop (a jump clears the fifo, hence the refill).
//
// Build with:
//   clang++ --target=aie4-none-unknown-elf -O2 -std=c++20
//
// Matrix multiply C[Mx64] += A[MxK] * B[Kx64] with MXINT8 packed inputs.
// MXINT8: S8 mantissa (6 frac bits) + E8M0 shared exponent, block_size=32
// (ebs32). M=4 rows, K=256 (4 outer iterations of 64), N=64 columns, fp32
// output. Per 128-element vector: 4 blocks of [1 SF + 32 mantissa] = 132 bytes.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

extern "C" {
void matmul(int8_t *restrict A, int8_t *restrict B, float *restrict C);
}

void matmul(int8_t *restrict A, int8_t *restrict B, float *restrict C) {
  constexpr unsigned K_outer = 4;
  constexpr unsigned M = 4;
  constexpr unsigned mxVecBytes = 132; // 128 int8 mantissa + 4 E8M0 (ebs32)

  v64accfloat acc[M];
#pragma clang loop unroll(full)
  for (unsigned l = 0; l < M; l++)
    acc[l] = undef_v64accfloat();

  fifo_state_t fA, fB;
  int zero_acc = 1;

  for (unsigned k = 0; k < K_outer; k++) {
    // B: 32 vectors (64 K x 64 N) through staging into the MMAC matrix.
    // Each vector gets its own fill + 1D (jump-mode) pop; see the header.
#pragma clang loop unroll(full)
    for (unsigned j = 0; j < 32; j++) {
      v128mxint8_ebs32_unaligned *restrict pB =
          (v128mxint8_ebs32_unaligned *)(B + (k * 32 + j) * mxVecBytes);
      fifo_ld_reset(pB, fB);
      fifo_ld_pop_1d_byte_staging(pB, fB, 0, j);
    }
    staging_to_matrix_m64x64mxint8();

    // A: M/2 vectors, each holding two rows of 64 K. Fill + 2D (jump-mode)
    // pop per vector; the 2D jump itself is a no-op here.
#pragma clang loop unroll(full)
    for (unsigned i = 0; i < M / 2; i++) {
      v128mxint8_ebs32_unaligned *restrict pA =
          (v128mxint8_ebs32_unaligned *)(A + (k * M / 2 + i) * mxVecBytes);
      addr_t cnt = 0;
      fifo_ld_reset(pA, fA);
      v128mxint8 a = fifo_ld_pop_2d_byte(pA, fA, 0, 0, cnt, 0);
      acc[i] = mac_conf(extract_v64mxint8(a, 0), acc[i], zero_acc);
      acc[i + M / 2] =
          mac_conf(extract_v64mxint8(a, 1), acc[i + M / 2], zero_acc);
    }
    zero_acc = 0;
  }

  v64accfloat *restrict pOut = (v64accfloat *)C;
#pragma clang loop unroll(full)
  for (unsigned l = 0; l < M / 2; l++) {
    *pOut++ = acc[l];
    *pOut++ = acc[l + M / 2];
  }
}

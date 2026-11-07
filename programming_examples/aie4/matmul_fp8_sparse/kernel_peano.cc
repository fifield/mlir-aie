//===- kernel_peano.cc - MMAC fp8*fp8 sparse matmul (Peano) ---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Peano (LLVM) port of kernel.cc: same entry points (matmul, matmul_hw),
// operand layouts and semantics; see kernel.cc and README.md for the sparse
// format.  Written with Peano builtins (no aie_api); aie4intrin.h is included
// implicitly by the aie4 target.
//
// Why a separate file: Peano cannot instruction-select any builtin that
// writes the sparsity register rows (zqa.r0..r5).  The two-argument
// load_staging(B, mask) used by kernel.cc, load_staging_sparse() and
// insert_staging_sparsity() all fail with "cannot select:
// G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.aie4.load.staging.*)", and
// function-body inline asm is not supported either.  So the data is staged
// with the one-argument load_staging(B), and the six 1-Kbit sparsity rows are
// loaded by a small helper written in file-scope assembly, exactly the
// `vldb zqa.rN` sequence chess emits for load_staging(B, mask).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

// void load_sparsity_rows(const uint32_t *mask): load the 768-byte (6-Kbit)
// sparsity register image into staging rows zqa.r0..r5 (pointer arg in p0).
// The trailing nops give the loads time to land before the caller's
// staging_to_matrix copy.  It uses no stack; the .stack_sizes entry (as
// -fstack-size-section emits for C functions) tells aiecc so.
asm(R"(
  .section .text.load_sparsity_rows,"ax",@progbits
  .globl load_sparsity_rows
  .type load_sparsity_rows,@function
  .p2align 4
load_sparsity_rows:
  vldb zqa.r0, [p0, #0x0]
  vldb zqa.r1, [p0, #0x80]
  vldb zqa.r2, [p0, #0x100]
  vldb zqa.r3, [p0, #0x180]
  vldb zqa.r4, [p0, #0x200]
  vldb zqa.r5, [p0, #0x280]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  ret lr
  nop
  nop
  nop
  nop
  nop
  .size load_sparsity_rows, .-load_sparsity_rows
  .section .stack_sizes,"o",@progbits,.text.load_sparsity_rows
  .long load_sparsity_rows
  .byte 0
  .text
)");

extern "C" {
void load_sparsity_rows(const uint32_t *mask);
void matmul(int8_t *__restrict A, int8_t *__restrict B, float *__restrict C,
            uint32_t *__restrict mask);
void matmul_hw(int8_t *__restrict AM, int8_t *__restrict B,
               float *__restrict C);
}

void matmul(int8_t *__restrict A, int8_t *__restrict B, float *__restrict C,
            uint32_t *__restrict mask) {
  constexpr unsigned int M = 4;

  // Stage B (64 stored rows x 64 columns of fp8, 32 x v128), then all six
  // sparsity rows, as chess does for load_staging(B, mask).
  load_staging((v128float8 *)B);
  load_sparsity_rows(mask);

  // Copy data + mask into the QF_Zb sparse matrix register.
  staging_to_matrix_m128x64float8_sparse();

  // MAC: each A row supplies its 128 dense fp8 activations.
  v64accfloat acc[M];
#pragma clang loop unroll(full)
  for (unsigned int i = 0; i < M; i++) {
    v128float8 Xbuff = *((v128float8 *)(A + i * 128));
    acc[i] = mac_conf(Xbuff, undef_v64accfloat(), /*zero_acc=*/1);
  }

  v64accfloat *__restrict pOut = (v64accfloat *)C;
#pragma clang loop unroll(full)
  for (unsigned l = 0; l < M; l++)
    *pOut++ = acc[l];
}

// Hardware (npu3) entry point: A (512 B) and the mask (768 B) arrive as one
// contiguous buffer because a compute tile has only two input DMA channels.
void matmul_hw(int8_t *__restrict AM, int8_t *__restrict B,
               float *__restrict C) {
  matmul(AM, B, C, (uint32_t *)(AM + 512));
}

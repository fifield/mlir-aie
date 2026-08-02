//===- kernel.cc - VMAC int32*int16 elementwise multiply ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Elementwise multiply of int32 * int16 vectors, producing int32 output (lower 32 bits).
// Uses AIE4 VMAC unit: 16 multipliers, 16 acc lanes, 64-bit accumulator.
// Output is shift-round-saturated from acc64 to int32 (shift=0).
// ISA: VMUL/VMAC with bmode=2.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

extern "C" {
void vmul_int32x16(int32_t *restrict a, int16_t *restrict b,
                   int32_t *restrict c);
}

void vmul_int32x16(int32_t *restrict a, int16_t *restrict b,
                   int32_t *restrict c) {
  constexpr int VECTOR_LEN = 16;
  constexpr int TOTAL_ELEMS = 256;

  for (int i = 0; i < TOTAL_ELEMS; i += VECTOR_LEN) {
    aie::vector<int32, VECTOR_LEN> va = aie::load_v<VECTOR_LEN>((int32_t *)(a + i));
    aie::vector<int16, VECTOR_LEN> vb = aie::load_v<VECTOR_LEN>((int16_t *)(b + i));
    aie::accum<acc64, VECTOR_LEN> acc = aie::mul(va, vb);
    aie::store_v((int32_t *)(c + i), acc.template to_vector<int32>(0));
  }
}

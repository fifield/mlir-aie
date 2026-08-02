//===- kernel.cc - VMAC int16*int16 elementwise multiply ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Elementwise multiply of two int16 vectors, producing int32 output.
// Uses AIE4 VMAC unit: 32 multipliers, 32 acc lanes, 32-bit accumulator.
// ISA: VMUL/VMAC instructions with bmode=1.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

extern "C" {
void vmul_int16(int16_t *restrict a, int16_t *restrict b, int32_t *restrict c);
}

void vmul_int16(int16_t *restrict a, int16_t *restrict b, int32_t *restrict c) {
  constexpr int VECTOR_LEN = 32;
  constexpr int TOTAL_ELEMS = 256;

  for (int i = 0; i < TOTAL_ELEMS; i += VECTOR_LEN) {
    aie::vector<int16, VECTOR_LEN> va = aie::load_v<VECTOR_LEN>((int16_t *)(a + i));
    aie::vector<int16, VECTOR_LEN> vb = aie::load_v<VECTOR_LEN>((int16_t *)(b + i));
    aie::accum<acc32, VECTOR_LEN> acc = aie::mul(va, vb);
    aie::store_v((int32_t *)(c + i), acc.template to_vector<int32_t>());
  }
}

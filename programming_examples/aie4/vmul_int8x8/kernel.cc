//===- kernel.cc - VMAC int8*int8 elementwise multiply --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Elementwise multiply of two int8 vectors, producing int32 output.
// Uses AIE4 VMAC unit: 32 multipliers, 32 acc lanes, 32-bit accumulator.
// ISA: VMUL/VMAC instructions with bmode=0.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

extern "C" {
void vmul_int8(int8_t *restrict a, int8_t *restrict b, int32_t *restrict c);
}

void vmul_int8(int8_t *restrict a, int8_t *restrict b, int32_t *restrict c) {
  // Process 32 elements at a time (VMAC has 32 int8 multiply lanes)
  constexpr int VECTOR_LEN = 32;
  constexpr int TOTAL_ELEMS = 256;

  for (int i = 0; i < TOTAL_ELEMS; i += VECTOR_LEN) {
    aie::vector<int8, VECTOR_LEN> va = aie::load_v<VECTOR_LEN>((int8_t *)(a + i));
    aie::vector<int8, VECTOR_LEN> vb = aie::load_v<VECTOR_LEN>((int8_t *)(b + i));
    aie::accum<acc32, VECTOR_LEN> acc = aie::mul(va, vb);
    aie::store_v((int32_t *)(c + i), acc.template to_vector<int32_t>());
  }
}

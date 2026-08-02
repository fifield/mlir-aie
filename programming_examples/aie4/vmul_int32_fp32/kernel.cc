//===- kernel.cc - VMAC int32*fp32 elementwise multiply ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Elementwise multiply of int32 * fp32 vectors, producing fp32 output.
// Uses AIE4 VMAC unit: 16 multipliers, 16 acc lanes, fp32 accumulator.
// ISA: VMUL.f/VMAC.f with amode=2 (int32 A operand, fp32 B operand).
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

extern "C" {
void vmul_int32_fp32(int32_t *restrict a, float *restrict b,
                     float *restrict c);
}

void vmul_int32_fp32(int32_t *restrict a, float *restrict b,
                     float *restrict c) {
  constexpr int VECTOR_LEN = 16;
  constexpr int TOTAL_ELEMS = 256;

  for (int i = 0; i < TOTAL_ELEMS; i += VECTOR_LEN) {
    aie::vector<int32, VECTOR_LEN> va = aie::load_v<VECTOR_LEN>((int32_t *)(a + i));
    aie::vector<float, VECTOR_LEN> vb = aie::load_v<VECTOR_LEN>((float *)(b + i));
    aie::accum<accfloat, VECTOR_LEN> acc = aie::mul(va, vb);
    aie::store_v((float *)(c + i), acc.template to_vector<float>());
  }
}

//===- kernel.cc - VMAC fp16*fp16 elementwise multiply --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Elementwise multiply of two fp16 vectors, producing fp32 output.
// Uses AIE4 VMAC unit: 32 multipliers, 32 acc lanes, fp32 accumulator.
// ISA: VMUL.f/VMAC.f with bmode=1.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

extern "C" {
void vmul_fp16(int16_t *restrict a, int16_t *restrict b, float *restrict c);
}

void vmul_fp16(int16_t *restrict a, int16_t *restrict b, float *restrict c) {
  constexpr int VECTOR_LEN = 32;
  constexpr int TOTAL_ELEMS = 256;

  for (int i = 0; i < TOTAL_ELEMS; i += VECTOR_LEN) {
    aie::vector<float16, VECTOR_LEN> va = aie::load_v<VECTOR_LEN>((float16 *)(a + i));
    aie::vector<float16, VECTOR_LEN> vb = aie::load_v<VECTOR_LEN>((float16 *)(b + i));
    aie::accum<accfloat, VECTOR_LEN> acc = aie::mul(va, vb);
    aie::store_v((float *)(c + i), acc.template to_vector<float>());
  }
}

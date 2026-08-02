//===- kernel.cc - VMAC bf16*bf16 64-lane MUL only ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Elementwise multiply of two bf16 vectors, producing bf16 output.
// Uses AIE4 VMAC unit in 64-lane mode: MUL only, no accumulate.
// ISA: VMUL.f.64 with bmode=2.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

extern "C" {
void vmul_bf16_64(bfloat16 *restrict a, bfloat16 *restrict b,
                  bfloat16 *restrict c);
}

void vmul_bf16_64(bfloat16 *restrict a, bfloat16 *restrict b,
                  bfloat16 *restrict c) {
  constexpr int VECTOR_LEN = 32;
  constexpr int TOTAL_ELEMS = 256;

  for (int i = 0; i < TOTAL_ELEMS; i += VECTOR_LEN) {
    aie::vector<bfloat16, VECTOR_LEN> va = aie::load_v<VECTOR_LEN>((bfloat16 *)(a + i));
    aie::vector<bfloat16, VECTOR_LEN> vb = aie::load_v<VECTOR_LEN>((bfloat16 *)(b + i));
    aie::accum<accfloat, VECTOR_LEN> acc = aie::mul(va, vb);
    aie::vector<bfloat16, VECTOR_LEN> result = acc.template to_vector<bfloat16>();
    aie::store_v((bfloat16 *)(c + i), result);
  }
}

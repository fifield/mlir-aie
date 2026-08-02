//===- kernel.cc - Vector add int16 ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

extern "C" {
void vadd_int16(int16_t *restrict a, int16_t *restrict b, int16_t *restrict c);
}

void vadd_int16(int16_t *restrict a, int16_t *restrict b, int16_t *restrict c) {
  constexpr int VECTOR_LEN = 32;
  constexpr int TOTAL_ELEMS = 256;

  for (int i = 0; i < TOTAL_ELEMS; i += VECTOR_LEN) {
    aie::vector<int16, VECTOR_LEN> va = aie::load_v<VECTOR_LEN>((int16_t *)(a + i));
    aie::vector<int16, VECTOR_LEN> vb = aie::load_v<VECTOR_LEN>((int16_t *)(b + i));
    aie::vector<int16, VECTOR_LEN> vc = aie::add(va, vb);
    aie::store_v((int16_t *)(c + i), vc);
  }
}

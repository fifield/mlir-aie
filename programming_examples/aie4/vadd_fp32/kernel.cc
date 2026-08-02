//===- kernel.cc - Vector add fp32 ----------------------------*- C++ -*-===//
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
void vadd_fp32(float *restrict a, float *restrict b, float *restrict c);
}

void vadd_fp32(float *restrict a, float *restrict b, float *restrict c) {
  constexpr int VECTOR_LEN = 16;
  constexpr int TOTAL_ELEMS = 256;

  for (int i = 0; i < TOTAL_ELEMS; i += VECTOR_LEN) {
    aie::vector<float, VECTOR_LEN> va = aie::load_v<VECTOR_LEN>((float *)(a + i));
    aie::vector<float, VECTOR_LEN> vb = aie::load_v<VECTOR_LEN>((float *)(b + i));
    aie::vector<float, VECTOR_LEN> vc = aie::add(va, vb);
    aie::store_v((float *)(c + i), vc);
  }
}

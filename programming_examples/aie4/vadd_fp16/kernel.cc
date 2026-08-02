//===- kernel.cc - Vector add fp16 ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Elementwise addition of two fp16 vectors, producing fp16 output.
// Uses AIE4 VADD.f instruction, 32 lanes.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

extern "C" {
void vadd_fp16(float16 *restrict a, float16 *restrict b,
               float16 *restrict c);
}

void vadd_fp16(float16 *restrict a, float16 *restrict b,
               float16 *restrict c) {
  constexpr int VECTOR_LEN = 32;
  constexpr int TOTAL_ELEMS = 256;

  for (int i = 0; i < TOTAL_ELEMS; i += VECTOR_LEN) {
    aie::vector<float16, VECTOR_LEN> va =
        aie::load_v<VECTOR_LEN>((float16 *)(a + i));
    aie::vector<float16, VECTOR_LEN> vb =
        aie::load_v<VECTOR_LEN>((float16 *)(b + i));
    aie::vector<float16, VECTOR_LEN> vc = aie::add(va, vb);
    aie::store_v((float16 *)(c + i), vc);
  }
}

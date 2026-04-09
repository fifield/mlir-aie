//===- vector_scalar_mul.cc -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <stdint.h>

extern "C" {

void vector_scalar_mul(int32_t *a_in, int32_t *c_out, int32_t *factor,
                       int32_t N) {
  event0();
  int32_t f = *factor;
  const aie::vector<int32_t, 16> vfact = aie::broadcast<int32_t, 16>(f);
  for (int i = 0; i < N; i += 16) {
    aie::vector<int32_t, 16> va = aie::load_v<16>(a_in + i);
    aie::accum<acc64, 16> acc = aie::mul(va, vfact);
    aie::vector<int32_t, 16> vc = acc.to_vector<int32_t>();
    aie::store_v(c_out + i, vc);
  }
  event1();
}

} // extern "C"

//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

#include "aie_api/aie.hpp"
#include <stdint.h>

static void test(v32int32 *input, v32int32 *output, int K) {

  for (int k = 0; k < K; k++)
    chess_prepare_for_pipelining {
      output[k] = me_primitive::get_ss(0) + input[k];
    }
}

extern "C" {
void kernel(v32int32 *input, v32int32 *output, int K);
}

void kernel(v32int32 *input, v32int32 *output, int K) {
  test(input, output, K);
}

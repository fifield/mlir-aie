//===- kernel.cc -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

// Simple passthrough kernel - copies input to output unchanged
extern "C" {

void passthrough(int32_t *restrict in, int32_t *restrict out) {
  for (int i = 0; i < 32; i++) {
    out[i] = in[i];
  }
}

} // extern "C"

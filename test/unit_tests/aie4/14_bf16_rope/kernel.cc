//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.
//
//===----------------------------------------------------------------------===//

extern "C" {
  void rope(bfloat16 *in,bfloat16 *sincos, bfloat16 *out);
}

void rope(bfloat16 *in, bfloat16 *sincos, bfloat16 *out) {
  for (int i=0;i<32;i++)
    out[i] = in[i] * sincos[i];
}


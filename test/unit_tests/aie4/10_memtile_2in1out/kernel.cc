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
void eltwise_add(int *a, int *b, int *c);
}

void eltwise_add(int *a, int *b, int *c) {

  for (int i = 0; i < 32; i++)
    c[i] = a[i] + b[i];
}

//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

extern "C" {
void func(int *a, int *b);
}

void func(int *a, int *b) {
  int x = a[12];
  int y = a[29];
  int z = x + y - 1;
  b[0] = z;
}

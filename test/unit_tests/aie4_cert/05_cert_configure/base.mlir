//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
// REQUIRES: dont_run
// RUN: echo FAIL | FileCheck %s
// CHECK: PASS

module {
  aie.device(npu3) {
    %tile_0_0 = aie.tile(0, 0) { elf_file = "cert_medusa.elf" }
  }
}
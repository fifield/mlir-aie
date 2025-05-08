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

aie.device(npu3) {
  aiex.runtime_sequence() {
    // col 1 row 2 lock 0
    aiex.npu.write32{ address=0x26AE000 : ui32, value=0x2 : ui32}
  }
}

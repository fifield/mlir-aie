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
  aie.runtime_sequence() {
    // col 0 row 2 lock 0
    aiex.npu.write32{ address=0x6AE000 : ui32, value=0x2 : ui32}
    aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
  }
}

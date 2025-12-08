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
  // The offset in the BD written here is 0x20
  memref.global "private" constant @blockwrite_data_1 : memref<9xi32> = dense<[0x0, 0x20, 0x100, 0x7000000, 0x10FF, 0x1, 0x1, 0x1, 0x400]>
  aie.runtime_sequence(%arg0: memref<256xi32>) {
    %2 = memref.get_global @blockwrite_data_1 : memref<9xi32>
    // Tile(0, 0) [0x18000] type:bd, bd_id:0, bd_word_id:0, direction:S2MM, channel:0
    aiex.npu.blockwrite(%2) {address = 0x18000 : ui32} : memref<9xi32>
    aiex.npu.address_patch {addr = 0x18004 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
    // Tile(0, 0) [0x18544] type:start_queue, direction:S2MM, channel:0
    aiex.npu.write32 {address = 0x18544 : ui32, value = 0x80000000 : ui32}
    aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
  }
}

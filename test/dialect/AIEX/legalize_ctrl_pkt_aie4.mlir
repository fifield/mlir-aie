//===- legalize_ctrl_pkt_aie4.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-legalize-ctrl-packet | FileCheck %s

// CHECK-LABEL: aie.device(npu3) {
// CHECK: aiex.control_packet {address = 7340032 : ui32, data = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>, opcode = 0 : i32, stream_id = 0 : i32}
// CHECK: aiex.control_packet {address = 7340032 : ui32, data = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>, opcode = 0 : i32, stream_id = 0 : i32}
// CHECK: aiex.control_packet {address = 7340096 : ui32, data = array<i32: 16>, opcode = 0 : i32, stream_id = 0 : i32}

aie.device(npu3) {
  aie.runtime_sequence() {
    aiex.control_packet {address = 7340032 : ui32, data = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 7340032 : ui32, data = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>, opcode = 0 : i32, stream_id = 0 : i32}
  }
}

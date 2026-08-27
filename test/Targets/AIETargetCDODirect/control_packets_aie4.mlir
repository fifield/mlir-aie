//===- control_packets_aie4.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-ctrlpkt-to-bin -aie-output-binary=false %s | FileCheck %s

// CHECK-NOT: 8000001D
// CHECK: 0000001D
// CHECK-NOT: 8DE2B808
// CHECK: 0DE2B808
// CHECK: 00000000
// CHECK: 00000001
// CHECK: 00000002
// CHECK: 00000003
// CHECK: 00000004
// CHECK: 00000005
// CHECK: 00000006
// CHECK: 00000007
// CHECK: 00000008
// CHECK: 00000009
// CHECK: 0000000A
// CHECK: 0000000B
// CHECK: 0000000C
// CHECK: 0000000D
// CHECK: 0000000E
// CHECK: 0000000F

module {
  aie.device(npu3) {
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    aie.runtime_sequence() {
      aiex.control_packet {address = 8052768 : ui32, data = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>, opcode = 2 : i32, stream_id = 1 : i32}
    }
  }
}

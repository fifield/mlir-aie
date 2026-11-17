//===- npu3_push_queue_repeat.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The AIE4 task word carries a 12-bit repeat count in [27:16], so a repeat
// count above 255 is not truncated to the AIE2 8-bit field.

// RUN: aie-opt --aie-dma-to-npu %s | FileCheck %s

// 0x812C0003: issue token, repeat 300, BD 3.
// CHECK: %[[CMD:.*]] = arith.constant -2127822845 : i32
// Shim MM2S 0 task queue: 0x18554 in column 1.
// CHECK: %[[ADDR:.*]] = arith.constant 33654100 : i32
// CHECK: aiex.npu.write32(%[[ADDR]], %[[CMD]])
module {
  aie.device(npu3) {
    %shim = aie.tile(1, 0)
    aie.runtime_sequence @seq() {
      %c3 = arith.constant 3 : i32
      %c300 = arith.constant 300 : i32
      aiex.npu.push_queue (1, 0, MM2S:0) bd_id %c3 repeat %c300 {issue_token = true} : i32, i32
    }
  }
}

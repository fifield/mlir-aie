//===- npu3_push_queue_bank.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu %s | FileCheck %s

// AIE4 shim Start_BD_ID is bank-local: BD 17 on MM2S 2 (bank 16-31) is 1.
// Repeat in [27:16]. MM2S_2 task queue 0x58554, MM2S_0 0x18554.
// CHECK-DAG: %[[V0:.*]] = arith.constant 131073 : i32
// CHECK-DAG: %[[A0:.*]] = arith.constant 361812 : i32
// CHECK: aiex.npu.write32(%[[A0]], %[[V0]]) : i32, i32
// CHECK-DAG: %[[V1:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[A1:.*]] = arith.constant 99668 : i32
// CHECK: aiex.npu.write32(%[[A1]], %[[V1]]) : i32, i32

module {
  aie.device(npu3) {
    aie.runtime_sequence() {
      %rc0 = arith.constant 2 : i32
      %bd0 = arith.constant 17 : i32
      aiex.npu.push_queue (0, 0, MM2S:2) bd_id %bd0 repeat %rc0 {issue_token = false} : i32, i32
      %rc1 = arith.constant 0 : i32
      %bd1 = arith.constant 5 : i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %bd1 repeat %rc1 {issue_token = false} : i32, i32
    }
  }
}

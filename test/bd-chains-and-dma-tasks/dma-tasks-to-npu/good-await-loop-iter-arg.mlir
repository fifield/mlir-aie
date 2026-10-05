//===- good-await-loop-iter-arg.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// Resolve an await on a loop iter_arg through its init and yielded tasks.
// The unrelated decoy uses a different tile, direction and channel so an
// incorrect resolution produces different sync operands.

// CHECK-LABEL: @await_loop_iter_arg
// CHECK: aiex.npu.sync
// CHECK: scf.for
// CHECK: aiex.npu.push_queue(2, 0, S2MM : 1)
// CHECK: %[[ROWS:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[COLS:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[CHAN:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[DIR:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[ROW:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[COL:.*]] = arith.constant 2 : i32
// CHECK: aiex.npu.sync(%[[COL]], %[[ROW]], %[[DIR]], %[[CHAN]], %[[COLS]], %[[ROWS]]) : i32, i32, i32, i32, i32, i32
// CHECK: }
// CHECK: %[[ROWS2:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[COLS2:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[CHAN2:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[DIR2:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[ROW2:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[COL2:.*]] = arith.constant 2 : i32
// CHECK: aiex.npu.sync(%[[COL2]], %[[ROW2]], %[[DIR2]], %[[CHAN2]], %[[COLS2]], %[[ROWS2]]) : i32, i32, i32, i32, i32, i32

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_2_0 = aie.tile(2, 0)
    aie.runtime_sequence @await_loop_iter_arg(%arg0: memref<1024xi32>, %n: index) {
      %decoy = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256) {bd_id = 2 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%decoy)
      aiex.dma_await_task(%decoy)

      %c1 = arith.constant 1 : index
      %init = aiex.dma_configure_task(%tile_2_0, S2MM, 1) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%init)
      %last = scf.for %i = %c1 to %n step %c1 iter_args(%prev = %init) -> (index) {
        %t = aiex.dma_configure_task(%tile_2_0, S2MM, 1) {
          aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 256) {bd_id = 1 : i32}
          aie.end
        } {issue_token = true}
        aiex.dma_start_task(%t)
        aiex.dma_await_task(%prev)
        scf.yield %t : index
      }
      aiex.dma_await_task(%last)
    }
  }
}

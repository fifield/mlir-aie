//===- good-await-if-in-loop-iter-arg.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// Resolve the loop's awaited iter_arg through both scf.if branches.
// All candidate tasks share a channel; the unrelated decoy uses a different
// tile, direction and channel so an incorrect resolution is visible.

// CHECK-LABEL: @await_if_in_loop_iter_arg
// CHECK: aiex.npu.sync
// CHECK: scf.for
// CHECK: %[[ROWS:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[COLS:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[CHAN:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[DIR:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[ROW:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[COL:.*]] = arith.constant 2 : i32
// CHECK: aiex.npu.sync(%[[COL]], %[[ROW]], %[[DIR]], %[[CHAN]], %[[COLS]], %[[ROWS]]) : i32, i32, i32, i32, i32, i32
// CHECK: scf.if
// CHECK: }

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_2_0 = aie.tile(2, 0)
    aie.runtime_sequence @await_if_in_loop_iter_arg(%arg0: memref<1024xi32>, %n: index, %cond: i1) {
      %decoy = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256) {bd_id = 3 : i32}
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
        aiex.dma_await_task(%prev)
        %t = scf.if %cond -> (index) {
          %ta = aiex.dma_configure_task(%tile_2_0, S2MM, 1) {
            aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 256) {bd_id = 1 : i32}
            aie.end
          } {issue_token = true}
          aiex.dma_start_task(%ta)
          scf.yield %ta : index
        } else {
          %tb = aiex.dma_configure_task(%tile_2_0, S2MM, 1) {
            aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 256) {bd_id = 2 : i32}
            aie.end
          } {issue_token = true}
          aiex.dma_start_task(%tb)
          scf.yield %tb : index
        }
        scf.yield %t : index
      }
      aiex.dma_await_task(%last)
    }
  }
}

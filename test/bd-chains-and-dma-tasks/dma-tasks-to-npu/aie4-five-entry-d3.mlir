// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --aie-dma-tasks-to-npu %s | FileCheck %s

// Both explicit forms of a four-dimensional AIE4 shim access reach D3: five
// entries with a unit iteration, and four entries plus the iteration attribute.

// CHECK-LABEL: @five_entry
// CHECK: aiex.npu.writebd
// CHECK-SAME: d2_size = 3 : i32
// CHECK-SAME: d3_size = 720 : i32, d3_stride = 5120 : i32
// CHECK-SAME: iteration_size = 0 : i32, iteration_stride = 0 : i32
module {
  aie.device(npu3) {
    %t = aie.tile(0, 0)
    aie.runtime_sequence @five_entry(%out: memref<11059200xi32>) {
      %0 = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%out : memref<11059200xi32> offset = 0 len = 5529600 sizes = [1, 720, 3, 2, 1280] strides = [0, 5120, 3686400, 2560, 1]) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}

// -----

// CHECK-LABEL: @four_entry_iteration_attr
// CHECK: aiex.npu.writebd
// CHECK-SAME: d2_size = 3 : i32
// CHECK-SAME: d3_size = 720 : i32, d3_stride = 5120 : i32
// CHECK-SAME: iteration_size = 0 : i32, iteration_stride = 0 : i32
module {
  aie.device(npu3) {
    %t = aie.tile(0, 0)
    aie.runtime_sequence @four_entry_iteration_attr(%out: memref<11059200xi32>) {
      %0 = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%out : memref<11059200xi32> offset = 0 len = 5529600 sizes = [720, 3, 2, 1280] strides = [5120, 3686400, 2560, 1]) {bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 1, stride = 0, current = 0>}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}

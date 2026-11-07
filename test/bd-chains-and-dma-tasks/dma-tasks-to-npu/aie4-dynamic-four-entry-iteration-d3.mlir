// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --aie-dma-tasks-to-npu --canonicalize %s | FileCheck %s

// A runtime offset takes the dynamic AIE4 shim lowering. There, as on the
// static path, an iteration attribute takes the iteration slot, so four
// entries are [d3, d2, d1, d0] and encode the same BD as the five-entry form.

// CHECK-LABEL: @five_entry
// CHECK: aiex.npu.blockwrite_values({{.*}}) values [[WORDS:.*]] : i32, i32
module {
  aie.device(npu3) {
    %t = aie.tile(0, 0)
    aie.runtime_sequence @five_entry(%out: memref<11059200xi32>, %off: i32) {
      %0 = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%out : memref<11059200xi32> offset = %off len = 30720 sizes = [1, 4, 3, 2, 1280] strides = [0, 5120, 3686400, 2560, 1]) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}

// -----

// CHECK-LABEL: @four_entry_iteration_attr
// CHECK: aiex.npu.blockwrite_values({{.*}}) values [[WORDS]] : i32, i32
module {
  aie.device(npu3) {
    %t = aie.tile(0, 0)
    aie.runtime_sequence @four_entry_iteration_attr(%out: memref<11059200xi32>, %off: i32) {
      %0 = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%out : memref<11059200xi32> offset = %off len = 30720 sizes = [4, 3, 2, 1280] strides = [5120, 3686400, 2560, 1]) {bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 1, stride = 0, current = 0>}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}

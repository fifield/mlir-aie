//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 AMD Inc.

// RUN: aie-opt --aie-assign-buffer-addresses --aie-dma-tasks-to-npu %s | FileCheck %s

// AIE4 (npu3) memtile equivalent of good-14: the explicit 5-dim access pattern
// lowers d3 as a real addressing dimension with iteration kept separate. Unlike
// AIE2, AIE4 encodes stepsizes as actual values (not actual-1) and has no D0
// stepsize (D0 is implied contiguous), so d0_stride lowers to 0.

module {
  aie.device(npu3) {
    %tile_0_1 = aie.tile(0, 1)
    %buf0 = aie.buffer(%tile_0_1) : memref<240xi32>

    aie.runtime_sequence() {
      %t0 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
          // CHECK: writebd
          // CHECK-SAME: d0_size = 4 : i32
          // CHECK-SAME: d0_stride = 0 : i32
          // CHECK-SAME: d1_size = 2 : i32
          // CHECK-SAME: d1_stride = 4 : i32
          // CHECK-SAME: d2_size = 2 : i32
          // CHECK-SAME: d2_stride = 8 : i32
          // CHECK-SAME: d3_size = 3 : i32
          // CHECK-SAME: d3_stride = 16 : i32
          // CHECK-SAME: iteration_size = 4 : i32
          // CHECK-SAME: iteration_stride = 48 : i32
          aie.dma_bd(%buf0 : memref<240xi32> offset = 0 len = 48 sizes = [5, 3, 2, 2, 4] strides = [48, 16, 8, 4, 1]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

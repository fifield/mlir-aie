//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 AMD Inc.

// RUN: aie-opt --aie-assign-buffer-addresses --aie-dma-tasks-to-npu %s | FileCheck %s

// MemTiles have a D3 addressing dimension on all generations. An explicit
// 5-dim access pattern [d0,d1,d2,d3,iteration] is lowered with d3 as a real
// addressing dimension and the iteration/repeat dimension kept separate, even
// on AIE2 (npu1). The lowest four dimensions (d0..d3) are addressing dims that
// together account for the BD transfer length (4*2*2*3 = 48 elements); the
// outermost dimension is the iteration/repeat.

module {
  aie.device(npu1) {
    %tile_0_1 = aie.tile(0, 1)
    %buf0 = aie.buffer(%tile_0_1) : memref<240xi32>

    aie.runtime_sequence() {
      %t0 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
          // CHECK: writebd
          // CHECK-SAME: d0_size = 4 : i32
          // CHECK-SAME: d1_size = 2 : i32
          // CHECK-SAME: d2_size = 2 : i32
          // CHECK-SAME: d3_size = 3 : i32
          // CHECK-SAME: d3_stride = 15 : i32
          // Iteration is kept separate from d3; AIE2 encodes the wrap as
          // (count - 1), so 5 iterations -> iteration_size = 4.
          // CHECK-SAME: iteration_size = 4 : i32
          // CHECK-SAME: iteration_stride = 47 : i32
          aie.dma_bd(%buf0 : memref<240xi32> offset = 0 len = 48 sizes = [5, 3, 2, 2, 4] strides = [48, 16, 8, 4, 1]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 AMD Inc.

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// AIE4 shim D3 addressing through the dma_configure_task lowering. Consistent
// with aiex.npu.dma_memcpy_nd, D3 is the 5th data dimension: an explicit 5-dim
// access pattern [d0,d1,d2,d3,iteration] lowers d3 as a real addressing
// dimension. All four addressing dimensions (d0..d3) contribute to the BD
// transfer length (4*3*2*2 = 48); the outermost dimension is iteration/repeat.
// A non-contiguous d3 stride keeps the transfer out of the linear fast path.

module {
  aie.device(npu3) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          // CHECK: writebd
          // CHECK-SAME: buffer_length = 48 : i32
          // CHECK-SAME: d0_size = 4 : i32
          // AIE4 D0 stride is fixed by hardware at 1 (d0_stride field = 0).
          // CHECK-SAME: d0_stride = 0 : i32
          // CHECK-SAME: d1_size = 3 : i32
          // CHECK-SAME: d1_stride = 4 : i32
          // CHECK-SAME: d2_size = 2 : i32
          // CHECK-SAME: d2_stride = 12 : i32
          // CHECK-SAME: d3_size = 2 : i32
          // AIE4 encodes stepsizes as actual values (not actual-1).
          // CHECK-SAME: d3_stride = 30 : i32
          // CHECK-SAME: iteration_size = 0 : i32
          // CHECK-SAME: iteration_stride = 0 : i32
          aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 48 sizes = [1, 2, 2, 3, 4] strides = [0, 30, 12, 4, 1]) {bd_id = 0 : i32}
          aie.end
      }
    }
    aie.shim_dma_allocation @a(%tile_0_0, MM2S, 0)
  }
}

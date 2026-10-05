//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defer channel validation until both tile coordinates are known, including
// for channels that will be rejected after placement.
// RUN: aie-opt --verify-diagnostics %s

module {
  aie.device(npu1) {
    %unknown = aie.logical_tile<ShimNOCTile>(?, ?)
    %column = aie.logical_tile<ShimNOCTile>(0, ?)
    %row = aie.logical_tile<ShimNOCTile>(?, 0)
    aie.shim_dma_allocation @alloc (%unknown, ControlMM2S, 0)
    aie.bd_chain @chain(%arg0: memref<64xi32>) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<64xi32>, %n: i64) {
      // Memcpy verification also defers channel and BD checks until placement,
      // before either the static or dynamic lowering path is selected.
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @alloc, id = 0 : i64} : memref<64xi32>
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 1, %n][0, 0, 0, 1]) {metadata = @alloc, id = 0 : i64} : memref<64xi32>
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @unresolved, id = 0 : i64} : memref<64xi32>
      %t0 = aiex.dma_configure_task(%unknown, ControlMM2S, 0) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      %t1 = aiex.dma_configure_task(%column, S2MM, 3) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      %t2 = aiex.dma_configure_task(%row, S2MM, 3) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      %t3 = aiex.dma_configure_task_for @alloc {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      %t4 = aiex.dma_start_bd_chain @chain(%arg0) : (memref<64xi32>) on (%unknown, ControlMM2S, 0)
      %t5 = aiex.dma_configure_task_for @unresolved {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
    }
  }
}

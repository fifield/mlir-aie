//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// xcvc1902 tile (0, 0) is a PL-only shim with no DMA engine. Reject the task
// before lowering queries its BD fields.

// RUN: aie-opt --verify-diagnostics --split-input-file --aie-dma-tasks-to-npu %s

module {
  aie.device(xcvc1902) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // expected-error@+1 {{unsupported DMA channel (MM2S, 0) on tile (0, 0)}}
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

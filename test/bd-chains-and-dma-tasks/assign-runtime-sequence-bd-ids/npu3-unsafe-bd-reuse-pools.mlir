//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids='warn-unreleased-transfers=false' \
// RUN:         --verify-diagnostics %s

// Unsafe reuse only warns within a pool: AIE4 memtile MM2S 1 id 0 is a
// different BD from MM2S 0 id 0.
aie.device(npu3) {
  %mem = aie.tile(0, 1)
  %buf = aie.buffer(%mem) : memref<1024xi32>
  aie.runtime_sequence @cross_pool() {
    %t0 = aiex.dma_configure_task(%mem, MM2S, 0) {
      aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t0)
    // expected-note@+1 {{released here}}
    aiex.dma_free_task(%t0)
    %t1 = aiex.dma_configure_task(%mem, MM2S, 1) {
      aie.dma_bd(%buf : memref<1024xi32> offset = 256 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t1)
    %t2 = aiex.dma_configure_task(%mem, MM2S, 0) {
      // expected-warning@+2 {{reuses buffer descriptor ID 0 on tile (0,1) after it was released by an aiex.dma_free_task that had no completion guarantee}}
      // expected-note@+1 {{not proven finished}}
      aie.dma_bd(%buf : memref<1024xi32> offset = 512 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t2)
  }
}

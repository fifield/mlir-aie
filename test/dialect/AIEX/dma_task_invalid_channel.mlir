//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Reject unsupported channels before BD allocation, including unassigned BDs.
// RUN: aie-opt --verify-diagnostics --split-input-file %s
// RUN: aie-opt --aie-substitute-shim-dma-allocations --aie-materialize-bd-chains --aie-assign-runtime-sequence-bd-ids --verify-diagnostics --split-input-file %s

// Reject control DMA on NPU2 before getBdPoolId can assert.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // expected-error@+1 {{unsupported DMA channel (ControlMM2S, 0) on tile (0, 0)}}
      %t = aiex.dma_configure_task(%tile, ControlMM2S, 0) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// AIE2 has no trace DMA engine.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // expected-error@+1 {{unsupported DMA channel (TraceS2MM, 0) on tile (0, 0)}}
      %t = aiex.dma_configure_task(%tile, TraceS2MM, 0) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// NPU1 shim data channels are limited to indices 0 and 1.
module {
  aie.device(npu1) {
    %tile = aie.tile(2, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // expected-error@+1 {{unsupported DMA channel (S2MM, 3) on tile (2, 0)}}
      %t = aiex.dma_configure_task(%tile, S2MM, 3) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// Negative channel indices are invalid.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // expected-error@+1 {{unsupported DMA channel (MM2S, 4294967295) on tile (0, 0)}}
      %t = aiex.dma_configure_task(%tile, MM2S, -1) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// AIE4 control DMA has only two channels.
module {
  aie.device(npu3) {
    %tile = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // expected-error@+1 {{unsupported DMA channel (ControlMM2S, 2) on tile (0, 0)}}
      %t = aiex.dma_configure_task(%tile, ControlMM2S, 2) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// AIE4 specialized DMA engines exist only on shim NOC tiles.
module {
  aie.device(npu3) {
    %tile = aie.tile(0, 2)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // expected-error@+1 {{unsupported DMA channel (ControlMM2S, 0) on tile (0, 2)}}
      %t = aiex.dma_configure_task(%tile, ControlMM2S, 0) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// Reject unsupported allocation channels before checking their task users.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 0)
    // expected-error@+1 {{unsupported DMA channel (TraceS2MM, 0) on tile (0, 0)}}
    aie.shim_dma_allocation @alloc (%tile, TraceS2MM, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task_for @alloc {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// Check explicit BD-chain channels before materialization.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 0)
    aie.bd_chain @chain(%arg0: memref<64xi32>) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // expected-error@+1 {{unsupported DMA channel (ControlMM2S, 0) on tile (0, 0)}}
      %t = aiex.dma_start_bd_chain @chain(%arg0) : (memref<64xi32>) on (%tile, ControlMM2S, 0)
    }
  }
}

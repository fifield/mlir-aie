//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Supported AIE4 control/trace channels must still reach their private BD
// pools. Ordinary shim MM2S channel 2 instead uses the upper shared pool.
// Cover direct tasks, allocation-based tasks, and materialized BD chains.
// RUN: aie-opt --aie-substitute-shim-dma-allocations --aie-materialize-bd-chains --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// CHECK: aiex.dma_configure_task(%{{.*}}, ControlMM2S, 1)
// CHECK: aie.dma_bd{{.*}}bd_id = 0 : i32
// CHECK: aiex.dma_configure_task(%{{.*}}, TraceS2MM, 0)
// CHECK: aie.dma_bd{{.*}}bd_id = 0 : i32
// CHECK: aiex.dma_configure_task(%{{.*}}, ControlMM2S, 0)
// CHECK: aie.dma_bd{{.*}}bd_id = 0 : i32
// CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 2)
// CHECK: aie.dma_bd{{.*}}bd_id = 16 : i32

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @trace (%shim, TraceS2MM, 0)
    aie.bd_chain @chain(%arg0: memref<64xi32>) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %ctrl = aiex.dma_configure_task(%shim, ControlMM2S, 1) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%ctrl)
      %trace = aiex.dma_configure_task_for @trace {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%trace)
      %chain = aiex.dma_start_bd_chain @chain(%arg0) : (memref<64xi32>) on (%shim, ControlMM2S, 0)
      %data = aiex.dma_configure_task(%shim, MM2S, 2) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%data)
    }
  }
}

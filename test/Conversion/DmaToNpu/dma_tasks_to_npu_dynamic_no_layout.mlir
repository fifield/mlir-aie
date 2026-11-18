//===- dma_tasks_to_npu_dynamic_no_layout.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu --verify-diagnostics %s

// A runtime-valued dma_bd packs its BD words through the target's BD register
// layout. AIE1 models none, even for a shim NOC tile, so the lowering rejects
// the BD rather than dereferencing the missing layout.

module {
  aie.device(xcvc1902) {
    %t = aie.tile(2, 0)
    aie.shim_dma_allocation @a(%t, MM2S, 0)
    aie.runtime_sequence @s(%arg0: memref<64xi32>, %n: i64) {
      %task = aiex.dma_configure_task_for @a {
        // expected-error@+1 {{'aie.dma_bd' op runtime-valued BD size/stride/len/bd_id/next_bd/lock value is only supported on tiles with a DMA buffer descriptor layout; use compile-time constants here.}}
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64 sizes = [1, 1, 4, %n] strides = [0, 0, 4, 1]) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%task)
      aiex.dma_free_task(%task)
    }
  }
}

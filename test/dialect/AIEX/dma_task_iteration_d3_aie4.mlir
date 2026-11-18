//===- dma_task_iteration_d3_aie4.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file %s

// No lowering programs the AIE4 shim d3 yet, so four dimensions plus the
// iteration attribute are refused.

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4096xi32>) {
      %t = aiex.dma_configure_task(%shim, MM2S, 0) {
        // expected-error @+1 {{Cannot give more than 3 dimensions for step sizes and wraps alongside the iteration attribute on this tile (got 4 dimensions): its fourth addressing dimension (d3) is not supported yet.}}
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 512 sizes = [2, 4, 8, 8] strides = [1024, 128, 16, 1]) {bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 2, stride = 2048, current = 0>}
        aie.end
      }
    }
  }
}

// -----

// Three dimensions and the iteration attribute fit.

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4096xi32>) {
      %t = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 256 sizes = [4, 8, 8] strides = [128, 16, 1]) {bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 2, stride = 2048, current = 0>}
        aie.end
      }
    }
  }
}

// -----

// Without the iteration attribute, the outermost dimension is the iteration.

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4096xi32>) {
      %t = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 512 sizes = [2, 4, 8, 8] strides = [1024, 128, 16, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

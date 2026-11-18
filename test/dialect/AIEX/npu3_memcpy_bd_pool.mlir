//===- npu3_memcpy_bd_pool.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// dma_memcpy_nd ids must lie in the channel's pool. AIE4 shim: MM2S 0-1 and
// S2MM 0 use BDs 0-15, MM2S 2-3 and S2MM 1 use BDs 16-31.

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @in(%shim, MM2S, 2)
  aie.runtime_sequence @seq(%arg0 : memref<256xi32>) {
    // expected-error@+1 {{'aiex.npu.dma_memcpy_nd' op programs buffer descriptor ID 3, which is not in the BD pool of MM2S channel 2 of tile (0, 0): ids 16 to 31}}
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 3 : i64, metadata = @in} : memref<256xi32>
  }
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @out(%shim, S2MM, 0)
  aie.runtime_sequence @seq(%arg0 : memref<256xi32>) {
    // expected-error@+1 {{'aiex.npu.dma_memcpy_nd' op programs buffer descriptor ID 20, which is not in the BD pool of S2MM channel 0 of tile (0, 0): ids 0 to 15}}
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 20 : i64, metadata = @out} : memref<256xi32>
  }
}

// -----

// Ids in the channel's bank are accepted.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @in(%shim, MM2S, 2)
  aie.shim_dma_allocation @out(%shim, S2MM, 0)
  aie.runtime_sequence @seq(%arg0 : memref<256xi32>, %arg1 : memref<256xi32>) {
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 17 : i64, metadata = @in} : memref<256xi32>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 3 : i64, metadata = @out} : memref<256xi32>
  }
}

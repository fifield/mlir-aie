// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --aie-dma-to-npu --verify-diagnostics %s

module {
 aie.device(npu3) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @input (%tile, MM2S, 0)
  aie.runtime_sequence @d2(%in: memref<1048576xi32>, %off: i64) {
   // expected-error@+1 {{d2 size hardware value 4096 exceeds hardware range [0:4095]}}
   aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0, %off][1, 2, 4096, 3, 4][0, 50000, 12, 4, 1]) {metadata = @input, id = 0 : i64} : memref<1048576xi32>
  }
 }
}

// -----
module {
 aie.device(npu3) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @input (%tile, MM2S, 0)
  aie.runtime_sequence @iter(%in: memref<1048576xi32>, %off: i64) {
   // expected-error@+1 {{iteration size hardware value 64 exceeds hardware range [0:63]}}
   aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0, %off][65, 2, 2, 3, 4][100, 30, 12, 4, 1]) {metadata = @input, id = 0 : i64} : memref<1048576xi32>
  }
 }
}

// -----
module {
 aie.device(npu3) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @input (%tile, MM2S, 0)
  aie.runtime_sequence @stride(%in: memref<1048576xi32>, %off: i64) {
   // expected-error@+1 {{stride 3 exceeds the hardware field range}}
   aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0, %off][1, 2, 2, 3, 4][0, 4194304, 12, 4, 1]) {metadata = @input, id = 0 : i64} : memref<1048576xi32>
  }
 }
}

// -----
module {
 aie.device(npu3) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @input (%tile, MM2S, 0)
  aie.runtime_sequence @d0(%in: memref<1048576xi32>, %off: i64) {
   // expected-error@+1 {{AIE4 D0 stride is fixed by hardware and must be 1}}
   aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0, %off][1, 2, 2, 3, 4][0, 30, 12, 4, 2]) {metadata = @input, id = 0 : i64} : memref<1048576xi32>
  }
 }
}

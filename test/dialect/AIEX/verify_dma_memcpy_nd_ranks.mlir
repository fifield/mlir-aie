//===- verify_dma_memcpy_nd_ranks.mlir -------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// The variadic access pattern allows 1..5 dimensions, and the offsets, sizes
// and strides lists must all agree in rank.

// Rank mismatch: 3 sizes but only 2 strides.
module {
  aie.device(npu1) {
    aie.runtime_sequence(%a: memref<64xi32>) {
      // expected-error @below {{Expected offsets, sizes and strides to have the same number of dimensions}}
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0][1, 8, 8][8, 1]) {metadata = @m, id = 0 : i64} : memref<64xi32>
    }
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @m(%t, MM2S, 0)
  }
}

// -----

// Too many dimensions: 6 entries exceeds the maximum of 5.
module {
  aie.device(npu1) {
    aie.runtime_sequence(%a: memref<64xi32>) {
      // expected-error @below {{expected at most 5 offset/size/stride dimensions}}
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0, 0, 0][1, 1, 1, 1, 8, 8][0, 0, 0, 0, 8, 1]) {metadata = @m, id = 0 : i64} : memref<64xi32>
    }
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @m(%t, MM2S, 0)
  }
}

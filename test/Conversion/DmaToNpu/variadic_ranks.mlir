//===- variadic_ranks.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-dma-to-npu %s | FileCheck %s

// aiex.npu.dma_memcpy_nd accepts a variadic 1..5 dimension access pattern. A
// shorter pattern is equivalent to a longer one padded with leading size-1
// no-op dimensions (see normalizeMemcpyNdDimVector). These tests pin down that
// backwards-compatible shortening behavior.

// A rank-1 access pattern [N] is a pure contiguous d0 transfer: buffer_length
// (the first BD word) is N, with no data-layout transformation.
// CHECK-LABEL: @rank1
// CHECK: memref.global "private" constant @[[G0:blockwrite_data_[0-9]+]] : memref<8xi32> = dense<[64,
// CHECK: get_global @[[G0]]
module @rank1 {
  aie.device(npu1) {
    aie.runtime_sequence(%a: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0][64][1]) {metadata = @m, id = 0 : i64} : memref<64xi32>
    }
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @m(%t, MM2S, 0)
  }
}

// -----

// Dropping leading size-1 dimensions is a no-op: a rank-3 [1, H, W] access
// pattern lowers to identical BD data as the explicit rank-4 [1, 1, H, W] form.
// Both transfers therefore share a single de-duplicated block-write global.
// CHECK-LABEL: @shorten
// CHECK: memref.global "private" constant @[[G1:blockwrite_data_[0-9]+]] : memref<8xi32> = dense<[32,
// CHECK: get_global @[[G1]]
// CHECK: get_global @[[G1]]
module @shorten {
  aie.device(npu1) {
    aie.runtime_sequence(%a: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0][1, 4, 8][0, 16, 1]) {metadata = @m, id = 0 : i64} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 4, 8][0, 0, 16, 1]) {metadata = @m, id = 1 : i64} : memref<64xi32>
    }
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @m(%t, MM2S, 0)
  }
}

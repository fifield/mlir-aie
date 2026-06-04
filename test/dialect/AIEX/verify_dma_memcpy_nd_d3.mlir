//===- verify_dma_memcpy_nd_d3.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

module {
  aie.device(npu2) {
    aie.runtime_sequence(%in : memref<128xi32>) {
      // expected-error @below {{D3 addressing is not supported on AIE2}}
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0, 0][1, 2, 2, 3, 4][0, 32, 12, 4, 1]) {
        metadata = @of_fromMem,
        id = 0 : i64
      } : memref<128xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem(%tile_0_0, MM2S, 0)
  }
}

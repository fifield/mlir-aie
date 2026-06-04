//===- dma_to_npu_stride_encoding.mlir -------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-dma-to-npu %s | FileCheck %s

// AIE4 shim encodes D1/D2/D3/iteration step sizes as actual 32-bit-word
// offsets. D0 has no step-size field and is encoded as the implied contiguous
// step.
// CHECK-LABEL: aie.device(npu3)
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<9xi32> = dense<[0, 0, 48, 50343938, 2097152, 5, 17, 4194404, 40960]>

module {
  aie.device(npu3) {
    aie.runtime_sequence(%in : memref<256xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0, 0][2, 2, 2, 3, 4][100, 40, 17, 5, 1]) {
        metadata = @of_fromMem,
        id = 0 : i64
      } : memref<256xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem(%tile_0_0, MM2S, 0)
  }
}

// -----

// AIE2 shim keeps the legacy actual-1 step-size encoding for D1/D2/iteration.
// The logical strides match the AIE4 case above, except D3 is inactive because
// AIE2 shims do not have a D3 addressing dimension.
// CHECK-LABEL: aie.device(npu2)
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[24, 0, 0, 4194304, -1070596092, 33554448, 1048623, 33554432]>

module {
  aie.device(npu2) {
    aie.runtime_sequence(%in : memref<256xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0, 0][2, 1, 2, 3, 4][48, 0, 17, 5, 1]) {
        metadata = @of_fromMem,
        id = 0 : i64
      } : memref<256xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem(%tile_0_0, MM2S, 0)
  }
}

// -----

// AIE4 has no D0_Stepsize field; the D0 stride is fixed at 1 by hardware. The
// resulting BD carries d0_stride as zero. Here only D1 is an active addressing
// dimension.
// CHECK-LABEL: aie.device(npu3)
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<9xi32> = dense<[0, 0, 3, 50343937, 524288, 4, 0, 0, 0]>

module {
  aie.device(npu3) {
    aie.runtime_sequence(%in : memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0, 0][1, 1, 1, 3, 1][0, 0, 0, 4, 1]) {
        metadata = @of_fromMem,
        id = 0 : i64
      } : memref<64xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem(%tile_0_0, MM2S, 0)
  }
}

// -----

// Sub-word contiguous D0 transfers are common for byte data. D0 is grouped into
// 32-bit words, and D1 still uses actual step-size encoding on AIE4.
// CHECK-LABEL: aie.device(npu3)
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<9xi32> = dense<[0, 0, 2, 50339841, 524288, 2, 0, 0, 0]>

module {
  aie.device(npu3) {
    aie.runtime_sequence(%in : memref<64xi8>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0, 0][1, 1, 1, 2, 4][0, 0, 0, 8, 1]) {
        metadata = @of_fromMem,
        id = 0 : i64
      } : memref<64xi8>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem(%tile_0_0, MM2S, 0)
  }
}

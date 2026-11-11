//===- dma_channel_valid.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The highest channel of each tile type is accepted, and a PLIO allocation
// on a shim PL tile names a stream port, not a DMA channel.

// RUN: aie-opt %s | FileCheck %s

// CHECK: aie.dma_start(S2MM, 5
// CHECK: aie.dma_start(MM2S, 5
// CHECK: aie.dma_start(MM2S, 1
// CHECK: aie.shim_dma_allocation @plio(%{{.*}}, MM2S, 0) {plio = true}
module {
  aie.device(xcve2802) {
    %t01 = aie.tile(2, 1)
    %t03 = aie.tile(2, 3)
    %pl = aie.tile(0, 0)
    %b01 = aie.buffer(%t01) : memref<16xi32>
    %b03 = aie.buffer(%t03) : memref<16xi32>
    aie.memtile_dma(%t01) {
      aie.dma_start(S2MM, 5, ^bd0, ^dma1)
    ^dma1:
      aie.dma_start(MM2S, 5, ^bd1, ^end)
    ^bd0:
      aie.dma_bd(%b01 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^bd1:
      aie.dma_bd(%b01 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.mem(%t03) {
      aie.dma_start(MM2S, 1, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b03 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.shim_dma_allocation @plio(%pl, MM2S, 0) {plio = true}
  }
}

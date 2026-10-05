// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --aie-assign-bd-ids %s | FileCheck %s

// Tile-wide reservations must only query supported channels, including the
// asymmetric AIE4 DMA counts, and reach each real data pool.
// CHECK-LABEL: module @core
// CHECK: aie.dma_bd
// CHECK-SAME: bd_id = 1 : i32
module @core {
  aie.device(npu3) {
    %tile = aie.tile(0, 2) {aiex.reserved_bd_ids = array<i32: 0>}
    %buf = aie.buffer(%tile) : memref<64xi32>
    aie.mem(%tile) {
      %start = aie.dma_start(S2MM, 1, ^bd, ^end)
    ^bd:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: module @memtile
// CHECK: aie.dma_bd
// CHECK-SAME: bd_id = 1 : i32
module @memtile {
  aie.device(npu3) {
    %tile = aie.tile(0, 1) {aiex.reserved_bd_ids = array<i32: 0>}
    %buf = aie.buffer(%tile) : memref<64xi32>
    aie.memtile_dma(%tile) {
      %start = aie.dma_start(MM2S, 9, ^bd, ^end)
    ^bd:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: module @shim
// CHECK: aie.dma_bd
// CHECK-SAME: bd_id = 17 : i32
module @shim {
  aie.device(npu3) {
    %tile = aie.tile(0, 0) {aiex.reserved_bd_ids = array<i32: 16>}
    %buf = aie.external_buffer : memref<64xi32>
    aie.shim_dma(%tile) {
      %start = aie.dma_start(MM2S, 3, ^bd, ^end)
    ^bd:
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

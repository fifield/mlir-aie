//===- npu3_bd_missing_channel.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids --verify-diagnostics --split-input-file %s

// aie.dma_start rejects a nonexistent channel before any BD id is checked. An
// AIE4 memtile has 10 MM2S and 8 S2MM channels.

aie.device(npu3) {
  %mem = aie.tile(0, 1)
  %buf = aie.buffer(%mem) {address = 0 : i32} : memref<256xi32>
  %dma = aie.memtile_dma(%mem) {
    // expected-error@+1 {{'aie.dma_start' op MM2S channel 10 is out of range: the tile has 10 MM2S channels}}
    %0 = aie.dma_start(MM2S, 10, ^bd0, ^end)
  ^bd0:
    aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

aie.device(npu3) {
  %mem = aie.tile(0, 1)
  %buf = aie.buffer(%mem) {address = 0 : i32} : memref<256xi32>
  %dma = aie.memtile_dma(%mem) {
    // expected-error@+1 {{'aie.dma_start' op S2MM channel 8 is out of range: the tile has 8 S2MM channels}}
    %0 = aie.dma_start(S2MM, 8, ^bd0, ^end)
  ^bd0:
    aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 3 : i32}
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

// The last channel of each direction is accessible.
aie.device(npu3) {
  %mem = aie.tile(0, 1)
  %buf = aie.buffer(%mem) {address = 0 : i32} : memref<256xi32>
  %dma = aie.memtile_dma(%mem) {
    %0 = aie.dma_start(MM2S, 9, ^bd0, ^in)
  ^in:
    %1 = aie.dma_start(S2MM, 7, ^bd1, ^end)
  ^bd0:
    aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
    aie.next_bd ^end
  ^bd1:
    aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

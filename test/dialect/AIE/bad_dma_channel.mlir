//===- bad_dma_channel.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A DMA channel index must name a channel the tile has in that direction, and
// a shim PL tile has no DMA. Without these checks the channel's registers are
// computed past the end of the channel's register block, e.g. memtile
// S2MM 6 programs the MM2S 0 control and queue registers.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu1) {
  %t = aie.tile(0, 1)
  %b = aie.buffer(%t) : memref<16xi32>
  aie.memtile_dma(%t) {
    // expected-error@+1 {{'aie.dma_start' op DMA channel (S2MM, 6) does not exist on tile (0, 1), which has 6 S2MM channel(s)}}
    aie.dma_start(S2MM, 6, ^bd0, ^end)
  ^bd0:
    aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

aie.device(npu2) {
  %t = aie.tile(0, 2)
  %b = aie.buffer(%t) : memref<16xi32>
  aie.mem(%t) {
    // expected-error@+1 {{'aie.dma_start' op DMA channel (MM2S, 2) does not exist on tile (0, 2), which has 2 MM2S channel(s)}}
    aie.dma_start(MM2S, 2, ^bd0, ^end)
  ^bd0:
    aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

aie.device(npu1) {
  %t = aie.tile(0, 0)
  aie.shim_dma(%t) {
    // expected-error@+1 {{'aie.dma_start' op DMA channel (S2MM, 2) does not exist on tile (0, 0), which has 2 S2MM channel(s)}}
    aie.dma_start(S2MM, 2, ^end, ^end)
  ^end:
    aie.end
  }
}

// -----

// The region form is checked too.
aie.device(npu1) {
  %t = aie.tile(0, 1)
  %b = aie.buffer(%t) : memref<16xi32>
  %l0 = aie.lock(%t, 0) {init = 1 : i32}
  %l1 = aie.lock(%t, 1) {init = 0 : i32}
  aie.memtile_dma(%t) {
    // expected-error@+1 {{'aie.dma' op DMA channel (S2MM, 6) does not exist on tile (0, 1), which has 6 S2MM channel(s)}}
    %0 = aie.dma(S2MM, 6) [{
      %c1 = arith.constant 1 : i32
      aie.use_lock(%l0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b : memref<16xi32>)
      aie.use_lock(%l1, Release, %c1)
    }]
    aie.end
  }
}

// -----

aie.device(npu1) {
  %t = aie.tile(0, 0)
  // expected-error@+1 {{'aie.shim_dma_allocation' op DMA channel (MM2S, 2) does not exist on tile (0, 0), which has 2 MM2S channel(s)}}
  aie.shim_dma_allocation @a(%t, MM2S, 2)
}

// -----

aie.device(npu1) {
  %t = aie.tile(0, 0)
  // expected-error@+1 {{'aie.shim_dma_allocation' op DMA channel (S2MM, -1) does not exist on tile (0, 0), which has 2 S2MM channel(s)}}
  aie.shim_dma_allocation @a(%t, S2MM, -1)
}

// -----

// The allocation index is 64-bit; a value that truncates to 0 is rejected.
aie.device(npu1) {
  %t = aie.tile(0, 0)
  // expected-error@+1 {{'aie.shim_dma_allocation' op DMA channel (MM2S, 4294967296) does not exist on tile (0, 0), which has 2 MM2S channel(s)}}
  aie.shim_dma_allocation @a(%t, MM2S, 4294967296)
}

// -----

// xcve2802 column 0 is a shim PL tile.
aie.device(xcve2802) {
  %t = aie.tile(0, 0)
  // expected-error@+1 {{'aie.shim_dma_allocation' op tile (0, 0) is a shim PL tile and has no DMA}}
  aie.shim_dma_allocation @a(%t, MM2S, 0)
}

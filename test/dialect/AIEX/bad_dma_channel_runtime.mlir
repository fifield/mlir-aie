//===- bad_dma_channel_runtime.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Runtime sequence DMA ops must name a DMA channel the tile has. Without the
// check, a push to shim S2MM 2 is lowered to a write of the MM2S 0 task queue.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu1) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<16xi32>) {
    // expected-error@+1 {{'aiex.dma_configure_task' op DMA channel (MM2S, 2) does not exist on tile (0, 0), which has 2 MM2S channel(s)}}
    %t1 = aiex.dma_configure_task(%t, MM2S, 2) {
      aie.dma_bd(%arg0 : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
      aie.end
    }
  }
}

// -----

aie.device(npu2) {
  %t = aie.tile(0, 1)
  %b = aie.buffer(%t) {address = 0 : i32} : memref<16xi32>
  aie.runtime_sequence(%arg0: memref<16xi32>) {
    // expected-error@+1 {{'aiex.dma_configure_task' op DMA channel (S2MM, 6) does not exist on tile (0, 1), which has 6 S2MM channel(s)}}
    %t1 = aiex.dma_configure_task(%t, S2MM, 6) {
      aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
      aie.end
    }
  }
}

// -----

aie.device(npu1) {
  aie.runtime_sequence(%arg0: memref<16xi32>) {
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{'aiex.npu.push_queue' op DMA channel (S2MM, 2) does not exist on tile (0, 0), which has 2 S2MM channel(s)}}
    aiex.npu.push_queue(0, 0, S2MM : 2) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
}

// -----

// xcve2802 column 0 is a shim PL tile.
aie.device(xcve2802) {
  aie.runtime_sequence(%arg0: memref<16xi32>) {
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{'aiex.npu.push_queue' op tile (0, 0) is a shim PL tile and has no DMA}}
    aiex.npu.push_queue(0, 0, S2MM : 0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
}

// -----

aie.device(npu1) {
  %t = aie.tile(0, 0)
  aie.bd_chain @c(%arg0: memref<16xi32>) {
    aie.dma_bd(%arg0 : memref<16xi32> offset = 0 len = 16)
    aie.end
  }
  aie.runtime_sequence(%arg0: memref<16xi32>) {
    // expected-error@+1 {{'aiex.dma_start_bd_chain' op DMA channel (MM2S, 3) does not exist on tile (0, 0), which has 2 MM2S channel(s)}}
    %t1 = aiex.dma_start_bd_chain @c(%arg0) : (memref<16xi32>) on (%t, MM2S, 3)
  }
}

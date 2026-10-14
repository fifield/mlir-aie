// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-dma-tasks-to-npu %s

// The AIE4 register database has Enable_Out_of_Order fields, but the target
// does not advertise SupportsOutOfOrderDMA. Fixing register lookup must not
// allow an out-of-order runtime task to reach channel-enable write emission.

module {
  aie.device(npu3) {
    %tile = aie.tile(0, 2)
    %buf = aie.buffer(%tile) {address = 0 : i32} : memref<8xi32>
    aie.runtime_sequence() {
      // expected-error@+1 {{out-of-order S2MM DMA is not supported on this device}}
      %task = aiex.dma_configure_task(%tile, S2MM, 0, <pkt_type = 0, pkt_id = 0>) {
        aie.dma_bd(%buf : memref<8xi32> offset = 0 len = 8) {bd_id = 0 : i32}
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%task)
    }
  }
}

// -----

module {
  aie.device(npu3) {
    %tile = aie.tile(0, 2)
    %buf = aie.buffer(%tile) {address = 0 : i32} : memref<8xi32>
    aie.runtime_sequence() {
      // expected-error@+1 {{out-of-order S2MM DMA is not supported on this device}}
      %task = aiex.dma_configure_task(%tile, S2MM, 1, <pkt_type = 0, pkt_id = 0>) {
        aie.dma_bd(%buf : memref<8xi32> offset = 0 len = 8) {bd_id = 0 : i32}
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%task)
    }
  }
}

// -----

module {
  aie.device(npu3) {
    %tile = aie.tile(0, 1)
    %buf = aie.buffer(%tile) {address = 0 : i32} : memref<8xi32>
    aie.runtime_sequence() {
      // expected-error@+1 {{out-of-order S2MM DMA is not supported on this device}}
      %task = aiex.dma_configure_task(%tile, S2MM, 0, <pkt_type = 0, pkt_id = 0>) {
        aie.dma_bd(%buf : memref<8xi32> offset = 0 len = 8) {bd_id = 0 : i32}
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%task)
    }
  }
}

//===- aie4-good-out-of-order-channel.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-dma-tasks-to-npu %s | FileCheck %s

// AIE4 S2MM channels support out-of-order mode in core, memory and shim tiles.
// An out_of_order runtime task sets Enable_Out_of_Order (bit 3, value 8, mask
// 8) in the channel's DMA_S2MM_<n>_Ctrl register. Addresses are the
// tile-local offsets from the AIE4 register database.

// Core tile (0,2) S2MM 0: memory-module DMA_S2MM_0_Ctrl at 0xACE00.
// CHECK-LABEL: @core_s2mm0
// CHECK-DAG: %[[ADDR:.*]] = arith.constant 708096 : i32
// CHECK-DAG: %[[VAL:.*]] = arith.constant 8 : i32
// CHECK-DAG: %[[MASK:.*]] = arith.constant 8 : i32
// CHECK: aiex.npu.maskwrite32(%[[ADDR]], %[[VAL]], %[[MASK]]) {column = 0 : i32, row = 2 : i32}
// CHECK: aiex.npu.writebd {{.*}}bd_id = 0 : i32{{.*}}enable_packet = 1 : i32{{.*}}row = 2 : i32
// CHECK: aiex.npu.push_queue(0, 2, S2MM : 0) {{.*}}issue_token = false
module @core_s2mm0 {
  aie.device(npu3) {
    %tile = aie.tile(0, 2)
    %buf = aie.buffer(%tile) {address = 0 : i32} : memref<8xi32>
    aie.runtime_sequence() {
      %task = aiex.dma_configure_task(%tile, S2MM, 0, <pkt_type = 0, pkt_id = 0>) {
        aie.dma_bd(%buf : memref<8xi32> offset = 0 len = 8) {bd_id = 0 : i32}
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%task)
    }
  }
}

// -----

// Core tile (0,2) S2MM 1: DMA_S2MM_1_Ctrl at 0xACE08.
// CHECK-LABEL: @core_s2mm1
// CHECK-DAG: %[[ADDR:.*]] = arith.constant 708104 : i32
// CHECK-DAG: %[[VAL:.*]] = arith.constant 8 : i32
// CHECK-DAG: %[[MASK:.*]] = arith.constant 8 : i32
// CHECK: aiex.npu.maskwrite32(%[[ADDR]], %[[VAL]], %[[MASK]]) {column = 0 : i32, row = 2 : i32}
// CHECK: aiex.npu.push_queue(0, 2, S2MM : 1)
module @core_s2mm1 {
  aie.device(npu3) {
    %tile = aie.tile(0, 2)
    %buf = aie.buffer(%tile) {address = 0 : i32} : memref<8xi32>
    aie.runtime_sequence() {
      %task = aiex.dma_configure_task(%tile, S2MM, 1, <pkt_type = 0, pkt_id = 0>) {
        aie.dma_bd(%buf : memref<8xi32> offset = 0 len = 8) {bd_id = 0 : i32}
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%task)
    }
  }
}

// -----

// Memory tile (0,1) S2MM 0: DMA_S2MM_0_Ctrl at 0x9E00. Receive BDs come from
// the channel's private pool, so their ids are local (0..15).
// CHECK-LABEL: @memtile_s2mm0
// CHECK-DAG: %[[ADDR:.*]] = arith.constant 40448 : i32
// CHECK-DAG: %[[VAL:.*]] = arith.constant 8 : i32
// CHECK-DAG: %[[MASK:.*]] = arith.constant 8 : i32
// CHECK: aiex.npu.maskwrite32(%[[ADDR]], %[[VAL]], %[[MASK]]) {column = 0 : i32, row = 1 : i32}
// CHECK: aiex.npu.push_queue(0, 1, S2MM : 0) {{.*}}issue_token = false
module @memtile_s2mm0 {
  aie.device(npu3) {
    %tile = aie.tile(0, 1)
    %buf = aie.buffer(%tile) {address = 0 : i32} : memref<8xi32>
    aie.runtime_sequence() {
      %task = aiex.dma_configure_task(%tile, S2MM, 0, <pkt_type = 0, pkt_id = 0>) {
        aie.dma_bd(%buf : memref<8xi32> offset = 0 len = 8) {bd_id = 0 : i32}
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%task)
    }
  }
}

// -----

// Memory tile (0,1) S2MM 3: DMA_S2MM_3_Ctrl at 0x9E18.
// CHECK-LABEL: @memtile_s2mm3
// CHECK-DAG: %[[ADDR:.*]] = arith.constant 40472 : i32
// CHECK: aiex.npu.maskwrite32(%[[ADDR]], %{{.*}}, %{{.*}}) {column = 0 : i32, row = 1 : i32}
module @memtile_s2mm3 {
  aie.device(npu3) {
    %tile = aie.tile(0, 1)
    %buf = aie.buffer(%tile) {address = 0 : i32} : memref<8xi32>
    aie.runtime_sequence() {
      %task = aiex.dma_configure_task(%tile, S2MM, 3, <pkt_type = 0, pkt_id = 0>) {
        aie.dma_bd(%buf : memref<8xi32> offset = 0 len = 8) {bd_id = 0 : i32}
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%task)
    }
  }
}

// -----

// Shim tile (0,0) S2MM 0: DMA_S2MM_0_Ctrl at 0x18540.
// CHECK-LABEL: @shim_s2mm0
// CHECK-DAG: %[[ADDR:.*]] = arith.constant 99648 : i32
// CHECK-DAG: %[[VAL:.*]] = arith.constant 8 : i32
// CHECK-DAG: %[[MASK:.*]] = arith.constant 8 : i32
// CHECK: aiex.npu.maskwrite32(%[[ADDR]], %[[VAL]], %[[MASK]]) {column = 0 : i32, row = 0 : i32}
module @shim_s2mm0 {
  aie.device(npu3) {
    %tile = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<8xi32>) {
      %task = aiex.dma_configure_task(%tile, S2MM, 0, <pkt_type = 0, pkt_id = 0>) {
        aie.dma_bd(%arg0 : memref<8xi32> offset = 0 len = 8) {bd_id = 0 : i32}
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%task)
    }
  }
}

// -----

// Shim tile (0,0) S2MM 1 uses the upper BD bank (BDs 16-31) and its own
// DMA_S2MM_1_Ctrl at 0x58540.
// CHECK-LABEL: @shim_s2mm1
// CHECK-DAG: %[[ADDR:.*]] = arith.constant 361792 : i32
// CHECK-DAG: %[[VAL:.*]] = arith.constant 8 : i32
// CHECK-DAG: %[[MASK:.*]] = arith.constant 8 : i32
// CHECK: aiex.npu.maskwrite32(%[[ADDR]], %[[VAL]], %[[MASK]]) {column = 0 : i32, row = 0 : i32}
// CHECK: aiex.npu.writebd {{.*}}bd_id = 16 : i32
module @shim_s2mm1 {
  aie.device(npu3) {
    %tile = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<8xi32>) {
      %task = aiex.dma_configure_task(%tile, S2MM, 1, <pkt_type = 0, pkt_id = 0>) {
        aie.dma_bd(%arg0 : memref<8xi32> offset = 0 len = 8) {bd_id = 16 : i32}
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%task)
    }
  }
}

// -----

// Out-of-order ignores Next_BD (Use_Next_BD is taken as 0), so the receive BDs
// of a multi-BD out-of-order task are written unchained, each releasing the
// completion lock alone.
// CHECK-LABEL: @memtile_multi_bd
// CHECK: aiex.npu.maskwrite32
// CHECK: aiex.npu.writebd {{.*}}bd_id = 0 : i32{{.*}}lock_acq_enable = 0 : i32{{.*}}next_bd = 0 : i32{{.*}}use_next_bd = 0 : i32
// CHECK: aiex.npu.writebd {{.*}}bd_id = 1 : i32{{.*}}lock_acq_enable = 0 : i32{{.*}}next_bd = 0 : i32{{.*}}use_next_bd = 0 : i32
// CHECK: aiex.npu.push_queue(0, 1, S2MM : 0) bd_id %{{.*}} repeat %c1_i32 {issue_token = false}
module @memtile_multi_bd {
  aie.device(npu3) {
    %tile = aie.tile(0, 1)
    %buf = aie.buffer(%tile) {address = 0 : i32} : memref<16xi32>
    %lock = aie.lock(%tile, 0) {init = 0 : i32}
    aie.runtime_sequence() {
      %task = aiex.dma_configure_task(%tile, S2MM, 0, <pkt_type = 0, pkt_id = 0>) {
        %c1 = arith.constant 1 : i32
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 8) {bd_id = 0 : i32}
        aie.use_lock(%lock, Release, %c1)
        aie.next_bd ^bd1
      ^bd1:
        %c1b = arith.constant 1 : i32
        aie.dma_bd(%buf : memref<16xi32> offset = 8 len = 8) {bd_id = 1 : i32}
        aie.use_lock(%lock, Release, %c1b)
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%task) {repeat_count = 1 : i32}
    }
  }
}

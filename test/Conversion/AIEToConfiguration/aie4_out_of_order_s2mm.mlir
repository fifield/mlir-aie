//===- aie4_out_of_order_s2mm.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-assign-bd-ids --convert-aie-to-transaction %s | FileCheck %s

// Static (load-time) out-of-order S2MM on AIE4 goes through aie-rt. The
// channel's DMA_S2MM_<n>_Ctrl gets Enable_Out_of_Order (bit 3), the task
// queue gets the packet count with no task-complete token, and the receive
// BDs are written with Use_Next_BD = 0 (hardware ignores Next_BD in this
// mode).

// Memory tile (0,1) is at 0x100000. Its receive BDs are the channel's private
// BDs 0 and 1; the last BD word holds Next_BD/Use_Next_BD and is 0.
// CHECK-LABEL: @memtile
// CHECK: memref.global "private" constant @config_blockwrite_data_{{[0-9]+}} : memref<11xi32> = dense<[{{.*}}, 0]>
// CHECK: memref.global "private" constant @config_blockwrite_data_{{[0-9]+}} : memref<11xi32> = dense<[{{.*}}, 0]>
// DMA_S2MM_1_Ctrl = 0x100000 + 0x9E08 = 1089032; DMA_S2MM_1_Start_Queue
// follows it, with repeat_count 1 in [27:16] and Enable_Token_Issue clear.
// CHECK: %[[CTRL:.*]] = arith.constant 1089032 : i32
// CHECK: %[[OOO:.*]] = arith.constant 8 : i32
// CHECK: aiex.npu.write32(%[[CTRL]], %[[OOO]])
// CHECK: %[[QUEUE:.*]] = arith.constant 1089036 : i32
// CHECK: %[[TASK:.*]] = arith.constant 65536 : i32
// CHECK: aiex.npu.write32(%[[QUEUE]], %[[TASK]])
module @memtile {
  aie.device(npu3) {
    %t = aie.tile(0, 1)
    %b0 = aie.buffer(%t) {address = 0 : i32} : memref<8xi32>
    %b1 = aie.buffer(%t) {address = 64 : i32} : memref<8xi32>
    %l = aie.lock(%t, 0) {init = 0 : i32}
    aie.memtile_dma(%t) {
      %0 = aie.dma_start(S2MM, 1, ^bd0, ^end, repeat_count = 1) {out_of_order}
    ^bd0:
      %c1 = arith.constant 1 : i32
      aie.dma_bd(%b0 : memref<8xi32> offset = 0 len = 8) {bd_id = 0 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(%l, Release, %c1)
      aie.next_bd ^bd1
    ^bd1:
      %c1b = arith.constant 1 : i32
      aie.dma_bd(%b1 : memref<8xi32> offset = 0 len = 8) {bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(%l, Release, %c1b)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}

// -----

// Core tile (0,2) is at 0x600000; DMA_S2MM_1_Ctrl is in its memory module at
// 0xACE08. BD word 5 holds Use_Next_BD [26]; only Lock_Rel_Value = 1 is set.
// CHECK-LABEL: @core
// CHECK: memref.global "private" constant @config_blockwrite_data_{{[0-9]+}} : memref<6xi32> = dense<[{{.*}}, 262144]>
// CHECK: memref.global "private" constant @config_blockwrite_data_{{[0-9]+}} : memref<6xi32> = dense<[{{.*}}, 262144]>
// CHECK: %[[CTRL:.*]] = arith.constant 6999560 : i32
// CHECK: %[[OOO:.*]] = arith.constant 8 : i32
// CHECK: aiex.npu.write32(%[[CTRL]], %[[OOO]])
// CHECK: %[[QUEUE:.*]] = arith.constant 6999564 : i32
// CHECK: %[[TASK:.*]] = arith.constant 65536 : i32
// CHECK: aiex.npu.write32(%[[QUEUE]], %[[TASK]])
module @core {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %b0 = aie.buffer(%t) {address = 0 : i32} : memref<8xi32>
    %b1 = aie.buffer(%t) {address = 64 : i32} : memref<8xi32>
    %l = aie.lock(%t, 0) {init = 0 : i32}
    aie.mem(%t) {
      %0 = aie.dma_start(S2MM, 1, ^bd0, ^end, repeat_count = 1) {out_of_order}
    ^bd0:
      %c1 = arith.constant 1 : i32
      aie.dma_bd(%b0 : memref<8xi32> offset = 0 len = 8) {bd_id = 0 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(%l, Release, %c1)
      aie.next_bd ^bd1
    ^bd1:
      %c1b = arith.constant 1 : i32
      aie.dma_bd(%b1 : memref<8xi32> offset = 0 len = 8) {bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(%l, Release, %c1b)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}

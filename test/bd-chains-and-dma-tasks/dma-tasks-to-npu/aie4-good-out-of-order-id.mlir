//===- aie4-good-out-of-order-id.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s --check-prefix=WRITEBD
// RUN: aie-opt --aie-dma-tasks-to-npu --aie-dma-to-npu %s | FileCheck %s --check-prefix=WORDS

// AIE4 sender BDs stamp the 4-bit out-of-order BD lookup index (0..15) that
// the MM2S channel inserts at [8:5] of the packet header. Each tile type packs
// it into its own BD field (AIE4 register database):
//   shim     DMA_BD<n>_8        Out_Of_Order_BD_ID [8:5]
//   memtile  DMA_MM2S_<c>_BD<n>_9 Out_Of_Order_BD_ID [3:0]
//   core     DMA_BD<n>_1        Out_Of_Order_BD_ID [27:24]

// WRITEBD: aiex.npu.writebd {{.*}}bd_id = 2 : i32{{.*}}out_of_order_id = 15 : i32{{.*}}row = 0 : i32
// WRITEBD: aiex.npu.writebd {{.*}}bd_id = 3 : i32{{.*}}out_of_order_id = 9 : i32{{.*}}row = 1 : i32
// WRITEBD: aiex.npu.writebd {{.*}}bd_id = 4 : i32{{.*}}out_of_order_id = 6 : i32{{.*}}row = 2 : i32

// Shim word 5 = Enable_Packet [31]; word 8 = (15 << 5) | pkt_id 1 = 481.
// WORDS: memref.global "private" constant @blockwrite_data_0 : memref<9xi32> = dense<[0, 0, 8, {{[0-9]+}}, 0, -2147483648, 0, 0, 481]>
// Memtile word 7 = Enable_Packet [21] | pkt_id 2 << 16 = 2228224; word 9 [3:0] = 9.
// WORDS: memref.global "private" constant @blockwrite_data_1 : memref<11xi32> = dense<[0, 8, {{-?[0-9]+}}, 0, 0, 0, 0, 2228224, 0, 9, 0]>
// Core word 1 = Enable_Packet [30] | 6 << 24 | pkt_id 3 << 19 = 1175977984.
// WORDS: memref.global "private" constant @blockwrite_data_2 : memref<6xi32> = dense<[8, 1175977984, 0, 0, 0, 0]>

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    %mbuf = aie.buffer(%mem) {address = 0 : i32} : memref<8xi32>
    %cbuf = aie.buffer(%core) {address = 0 : i32} : memref<8xi32>
    aie.runtime_sequence(%arg0: memref<8xi32>) {
      %t0 = aiex.dma_configure_task(%shim, MM2S, 0, <pkt_type = 0, pkt_id = 1>) {
        aie.dma_bd(%arg0 : memref<8xi32> offset = 0 len = 8) {bd_id = 2 : i32, out_of_order_id = 15 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task(%mem, MM2S, 0, <pkt_type = 0, pkt_id = 2>) {
        aie.dma_bd(%mbuf : memref<8xi32> offset = 0 len = 8) {bd_id = 3 : i32, out_of_order_id = 9 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task(%core, MM2S, 0, <pkt_type = 0, pkt_id = 3>) {
        aie.dma_bd(%cbuf : memref<8xi32> offset = 0 len = 8) {bd_id = 4 : i32, out_of_order_id = 6 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)
    }
  }
}

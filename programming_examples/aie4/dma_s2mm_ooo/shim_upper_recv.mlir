//===- shim_upper_recv.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NPU3 out-of-order S2MM into the upper BD bank of a shim tile. Shim S2MM 1 can
// only use BDs 16-31, but the packet header's BD lookup index is 4 bits. Like
// Next_BD and the task's start BD, the index is bank-local: index k selects
// BD 16 + k. Four MM2S senders in columns 1 and 2 merge into shim (0,0)
// S2MM 1; sender s stamps index (s + 1) % 4, so its packet lands in
// BD 16 + (s + 1) % 4, i.e. slot (s + 1) % 4 of the output.
//
// An out-of-order task cannot issue a completion token, so each receive BD
// releases the shim lock `cons`. A token-issuing S2MM 0 task acquires
// `cons` >= 4 before receiving an 8-word sentinel looped back from shim
// (0,0) MM2S 0; awaiting that task waits for all four packets.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu3) {
    %shim_0_0 = aie.tile(0, 0)
    %shim_1_0 = aie.tile(1, 0)
    %shim_2_0 = aie.tile(2, 0)

    %cons = aie.lock(%shim_0_0, 0) {init = 0 : i32, sym_name = "cons"}
    %cons_done = aie.lock(%shim_0_0, 1) {init = 0 : i32, sym_name = "cons_done"}

    aie.packet_flow(0) {
      aie.packet_source<%shim_1_0, DMA : 0>
      aie.packet_dest<%shim_0_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(1) {
      aie.packet_source<%shim_1_0, DMA : 1>
      aie.packet_dest<%shim_0_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(2) {
      aie.packet_source<%shim_2_0, DMA : 0>
      aie.packet_dest<%shim_0_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(3) {
      aie.packet_source<%shim_2_0, DMA : 1>
      aie.packet_dest<%shim_0_0, DMA : 1>
    } {keep_pkt_header = true}

    aie.flow(%shim_0_0, DMA : 0, %shim_0_0, DMA : 0)

    aie.runtime_sequence @sequence(%input: memref<264xi32>, %output: memref<264xi32>) {
      %recv = aiex.dma_configure_task(%shim_0_0, S2MM, 1, <pkt_type = 0, pkt_id = 0>) {
        %c1 = arith.constant 1 : i32
        aie.dma_bd(%output : memref<264xi32> offset = 0 len = 64) {bd_id = 16 : i32}
        aie.use_lock(%cons, Release, %c1)
        aie.next_bd ^bd1
      ^bd1:
        %c1b = arith.constant 1 : i32
        aie.dma_bd(%output : memref<264xi32> offset = 64 len = 64) {bd_id = 17 : i32}
        aie.use_lock(%cons, Release, %c1b)
        aie.next_bd ^bd2
      ^bd2:
        %c1c = arith.constant 1 : i32
        aie.dma_bd(%output : memref<264xi32> offset = 128 len = 64) {bd_id = 18 : i32}
        aie.use_lock(%cons, Release, %c1c)
        aie.next_bd ^bd3
      ^bd3:
        %c1d = arith.constant 1 : i32
        aie.dma_bd(%output : memref<264xi32> offset = 192 len = 64) {bd_id = 19 : i32}
        aie.use_lock(%cons, Release, %c1d)
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%recv) {repeat_count = 3 : i32}

      // Completion: wait for the four receive BDs, then take the sentinel.
      %tok = aiex.dma_configure_task(%shim_0_0, S2MM, 0) {
        %c4 = arith.constant 4 : i32
        %c1 = arith.constant 1 : i32
        aie.use_lock(%cons, AcquireGreaterEqual, %c4)
        aie.dma_bd(%output : memref<264xi32> offset = 256 len = 8) {bd_id = 0 : i32}
        aie.use_lock(%cons_done, Release, %c1)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tok)
      %sentinel = aiex.dma_configure_task(%shim_0_0, MM2S, 0) {
        aie.dma_bd(%input : memref<264xi32> offset = 256 len = 8) {bd_id = 1 : i32}
        aie.end
      }
      aiex.dma_start_task(%sentinel)

      %s0 = aiex.dma_configure_task(%shim_1_0, MM2S, 0, <pkt_type = 0, pkt_id = 0>) {
        aie.dma_bd(%input : memref<264xi32> offset = 0 len = 64) {bd_id = 0 : i32, out_of_order_id = 1 : i32}
        aie.end
      }
      %s1 = aiex.dma_configure_task(%shim_1_0, MM2S, 1, <pkt_type = 0, pkt_id = 1>) {
        aie.dma_bd(%input : memref<264xi32> offset = 64 len = 64) {bd_id = 1 : i32, out_of_order_id = 2 : i32}
        aie.end
      }
      %s2 = aiex.dma_configure_task(%shim_2_0, MM2S, 0, <pkt_type = 0, pkt_id = 2>) {
        aie.dma_bd(%input : memref<264xi32> offset = 128 len = 64) {bd_id = 0 : i32, out_of_order_id = 3 : i32}
        aie.end
      }
      %s3 = aiex.dma_configure_task(%shim_2_0, MM2S, 1, <pkt_type = 0, pkt_id = 3>) {
        aie.dma_bd(%input : memref<264xi32> offset = 192 len = 64) {bd_id = 1 : i32, out_of_order_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%s3)
      aiex.dma_start_task(%s2)
      aiex.dma_start_task(%s1)
      aiex.dma_start_task(%s0)

      aiex.dma_await_task(%tok)
    }
  }
}

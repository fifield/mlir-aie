//===- memtile_recv.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NPU3 out-of-order S2MM into a memory tile: four shim MM2S senders (two
// channels in each of columns 0 and 1) merge into MEM (0,1) S2MM 0, which is in
// out-of-order mode. Sender s stamps BD lookup index (s + 1) % 4, so its packet
// lands in receive BD (s + 1) % 4, i.e. slot (s + 1) % 4 of the merged buffer,
// regardless of arrival order. Each receive BD releases `full`; MEM MM2S 0
// waits for all four and drains the merged buffer to the host.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu3) {
    %shim_0_0 = aie.tile(0, 0)
    %shim_1_0 = aie.tile(1, 0)
    %mem_0_1 = aie.tile(0, 1)

    %merged = aie.buffer(%mem_0_1) {sym_name = "merged"} : memref<256xi32>
    %full = aie.lock(%mem_0_1, 0) {init = 0 : i32, sym_name = "full"}
    %done = aie.lock(%mem_0_1, 1) {init = 0 : i32, sym_name = "done"}

    // The receiving S2MM consumes the packet header, so keep it on the route.
    aie.packet_flow(0) {
      aie.packet_source<%shim_0_0, DMA : 0>
      aie.packet_dest<%mem_0_1, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(1) {
      aie.packet_source<%shim_0_0, DMA : 1>
      aie.packet_dest<%mem_0_1, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(2) {
      aie.packet_source<%shim_1_0, DMA : 0>
      aie.packet_dest<%mem_0_1, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(3) {
      aie.packet_source<%shim_1_0, DMA : 1>
      aie.packet_dest<%mem_0_1, DMA : 0>
    } {keep_pkt_header = true}

    aie.flow(%mem_0_1, DMA : 0, %shim_0_0, DMA : 0)

    aie.runtime_sequence @sequence(%input: memref<256xi32>, %output: memref<256xi32>) {
      // Arm the out-of-order receiver first: one receive BD per lookup index,
      // and a task that completes after four packets.
      %recv = aiex.dma_configure_task(%mem_0_1, S2MM, 0, <pkt_type = 0, pkt_id = 0>) {
        %c1 = arith.constant 1 : i32
        aie.dma_bd(%merged : memref<256xi32> offset = 0 len = 64) {bd_id = 0 : i32}
        aie.use_lock(%full, Release, %c1)
        aie.next_bd ^bd1
      ^bd1:
        %c1b = arith.constant 1 : i32
        aie.dma_bd(%merged : memref<256xi32> offset = 64 len = 64) {bd_id = 1 : i32}
        aie.use_lock(%full, Release, %c1b)
        aie.next_bd ^bd2
      ^bd2:
        %c1c = arith.constant 1 : i32
        aie.dma_bd(%merged : memref<256xi32> offset = 128 len = 64) {bd_id = 2 : i32}
        aie.use_lock(%full, Release, %c1c)
        aie.next_bd ^bd3
      ^bd3:
        %c1d = arith.constant 1 : i32
        aie.dma_bd(%merged : memref<256xi32> offset = 192 len = 64) {bd_id = 3 : i32}
        aie.use_lock(%full, Release, %c1d)
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%recv) {repeat_count = 3 : i32}

      // Drain once all four packets have landed.
      %drain = aiex.dma_configure_task(%mem_0_1, MM2S, 0) {
        %c4 = arith.constant 4 : i32
        %c1 = arith.constant 1 : i32
        aie.use_lock(%full, AcquireGreaterEqual, %c4)
        aie.dma_bd(%merged : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
        aie.use_lock(%done, Release, %c1)
        aie.end
      }
      aiex.dma_start_task(%drain)

      %out = aiex.dma_configure_task(%shim_0_0, S2MM, 0) {
        aie.dma_bd(%output : memref<256xi32> offset = 0 len = 256) {bd_id = 2 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%out)

      // Senders: sender s sends input[64*s, 64*s + 64) as packet s with BD
      // lookup index (s + 1) % 4.
      %s0 = aiex.dma_configure_task(%shim_0_0, MM2S, 0, <pkt_type = 0, pkt_id = 0>) {
        aie.dma_bd(%input : memref<256xi32> offset = 0 len = 64) {bd_id = 0 : i32, out_of_order_id = 1 : i32}
        aie.end
      }
      %s1 = aiex.dma_configure_task(%shim_0_0, MM2S, 1, <pkt_type = 0, pkt_id = 1>) {
        aie.dma_bd(%input : memref<256xi32> offset = 64 len = 64) {bd_id = 1 : i32, out_of_order_id = 2 : i32}
        aie.end
      }
      %s2 = aiex.dma_configure_task(%shim_1_0, MM2S, 0, <pkt_type = 0, pkt_id = 2>) {
        aie.dma_bd(%input : memref<256xi32> offset = 128 len = 64) {bd_id = 0 : i32, out_of_order_id = 3 : i32}
        aie.end
      }
      %s3 = aiex.dma_configure_task(%shim_1_0, MM2S, 1, <pkt_type = 0, pkt_id = 3>) {
        aie.dma_bd(%input : memref<256xi32> offset = 192 len = 64) {bd_id = 1 : i32, out_of_order_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%s3)
      aiex.dma_start_task(%s2)
      aiex.dma_start_task(%s1)
      aiex.dma_start_task(%s0)

      aiex.dma_await_task(%out)
    }
  }
}

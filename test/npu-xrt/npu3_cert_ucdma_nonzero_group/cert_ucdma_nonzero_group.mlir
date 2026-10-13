//===- cert_ucdma_nonzero_group.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Device-level chains and their shared payloads must be emitted in group 2,
// which executes on column 1's controller. The host reads the actual DMA
// destinations in tile(1,2), including guards initialized by device config.
module {
  aie.device(npu3) {
    memref.global "private" constant @head : memref<8xi32> = dense<[
      0x11110001, 0x11110002, 0x11110003, 0x11110004,
      0x11110005, 0x11110006, 0x11110007, 0x11110008]>
    memref.global "private" constant @tail : memref<8xi32> = dense<[
      0x22220001, 0x22220002, 0x22220003, 0x22220004,
      0x22220005, 0x22220006, 0x22220007, 0x22220008]>

    // tile(1,2) has array base 0x02600000; obuf starts at offset 0x400.
    aiex.cert.uc_dma_chain @sync_chain {
      aiex.cert.uc_dma_bd @head, 0x02600400, 8, true
      aiex.cert.uc_dma_bd @tail, 0x02600420, 8, false
    }
    aiex.cert.uc_dma_chain @async_chain {
      aiex.cert.uc_dma_bd @head, 0x02600480, 8, true
      aiex.cert.uc_dma_bd @tail, 0x026004A0, 8, false
    }

    %shim0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %shim1 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
    %tile = aie.tile(1, 2)
    aie.packet_flow(0) {
      aie.packet_source<%shim0, Control32 : 0>
      aie.packet_dest<%shim0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(1) {
      aie.packet_source<%shim1, Control32 : 0>
      aie.packet_dest<%shim1, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}

    %obuf = aie.buffer(%tile) {sym_name = "obuf", address = 1024 : i32} : memref<64xi32> = dense<0x0BADF00D>
    %go = aie.lock(%tile, 0) {init = 0 : i32}
    %done = aie.lock(%tile, 1) {init = 0 : i32}
    %one = arith.constant 1 : i32
    aie.mem(%tile) {
      %start = aie.dma_start(MM2S, 0, ^transfer, ^end)
    ^transfer:
      aie.use_lock(%go, AcquireGreaterEqual, %one)
      aie.dma_bd(%obuf : memref<64xi32> offset = 0 len = 64)
      aie.use_lock(%done, Release, %one)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.flow(%tile, DMA : 0, %shim1, DMA : 0)
    aie.shim_dma_allocation @out (%shim1, S2MM, 0)

    aiex.cert.attach_to_group(2) {
      aiex.cert.job(7) {
        // Wait until group 0 has initialized memory, locks, and the shim DMA.
        aiex.cert.remote_barrier(rb1, 0x5)
        aiex.cert.uc_dma_write_des_sync(@sync_chain)
        aiex.cert.uc_dma_write_des(0, @async_chain)
        aiex.cert.wait_uc_dma(0)
        // Release tile(1,2)'s lock 0 only after both payloads have landed.
        aiex.cert.write32(0x026AE000, 1)
        // Column 1's completion token is consumed by its own controller.
        aiex.cert.wait_tcts(32, 0, 1)
        aiex.cert.remote_barrier(rb2, 0x5)
      }
    }

    aie.runtime_sequence @sequence(%out: memref<64xi32>) {
      %task = aiex.dma_configure_task_for @out {
        aie.dma_bd(%out : memref<64xi32> offset = 0 len = 64) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%task)
    }
    aiex.cert.attach_to_group(0) {
      aiex.cert.job(8) {
        aiex.cert.remote_barrier(rb1, 0x5)
        // Keep the host invocation active until group 2 finishes readback.
        aiex.cert.remote_barrier(rb2, 0x5)
      }
    }
  }
}

//===- cert_multi_uc.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
module {
  aie.device(npu3) {
    %tile_0_0 = aie.tile(0, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %tile_1_0 = aie.tile(1, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}

    %tile_0_3 = aie.tile(0, 3)
    %tile_1_2 = aie.tile(1, 2)

    // Packet flows: each column routes to its own controller
    aie.packet_flow(0) {
      aie.packet_source<%tile_0_0, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(1) {
      aie.packet_source<%tile_1_0, Control32 : 0>
      aie.packet_dest<%tile_1_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}

    // Locks for CERT synchronization - init to 0, CERT will write 1 to release
    // Using lock 5 to avoid conflicts with objectfifo-generated locks
    %lock_0_3_5 = aie.lock(%tile_0_3, 5) { init = 0 : i32 }
    %lock_1_2_5 = aie.lock(%tile_1_2, 5) { init = 0 : i32 }

    // Output objectfifos: each core sends data to its column's shim tile
    aie.objectfifo @outC0 (%tile_0_3, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @outC1 (%tile_1_2, {%tile_1_0}, 2 : i32) : !aie.objectfifo<memref<32xi32>>

    %core_0_3 = aie.core(%tile_0_3) {
      // Wait for CERT to release the lock
      %c2_lock = arith.constant 2 : i32
      aie.use_lock(%lock_0_3_5, AcquireGreaterEqual, %c2_lock)
      // Produce output data: values 0..31
      %elem = aie.objectfifo.acquire @outC0 (Produce, 1) : memref<32xi32>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      scf.for %i = %c0 to %c32 step %c1 {
        %val = arith.index_cast %i : index to i32
        memref.store %val, %elem[%i] : memref<32xi32>
      }
      aie.objectfifo.release @outC0 (Produce, 1)
      aie.end
    }

    %core_1_2 = aie.core(%tile_1_2) {
      // Wait for CERT to release the lock
      %c2_lock = arith.constant 2 : i32
      aie.use_lock(%lock_1_2_5, AcquireGreaterEqual, %c2_lock)
      // Produce output data: values 100..131
      %elem = aie.objectfifo.acquire @outC1 (Produce, 1) : memref<32xi32>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %offset = arith.constant 100 : i32
      scf.for %i = %c0 to %c32 step %c1 {
        %idx = arith.index_cast %i : index to i32
        %val = arith.addi %idx, %offset : i32
        memref.store %val, %elem[%i] : memref<32xi32>
      }
      aie.objectfifo.release @outC1 (Produce, 1)
      aie.end
    }

    // Column 1 CERT job
    aiex.cert.attach_to_group(2) {
      aiex.cert.job(1) {
        // Without the barrier, configuration of column 1 locks by column 0
        // will race with the write32
        aiex.cert.remote_barrier (rb1, 0x5)
        // col 1 row 2 lock 5 - release the core
        %addr_c1 = arith.constant 0x26AE050 : i32
        %val_c1 = arith.constant 0x2 : i32
        aiex.npu.write32(%addr_c1, %val_c1) : i32, i32
        aiex.cert.wait_tcts(32, 0, 1)
        aiex.cert.remote_barrier (rb2, 0x5)
      }
    }

    // Runtime sequence: configure shim DMA to capture core outputs
    aie.runtime_sequence @sequence(%out : memref<64xi32>) {
      %t0 = aiex.dma_configure_task_for @outC0 {
        aie.dma_bd(%out : memref<64xi32> offset = 0 len = 32)
        aie.end
      } {issue_token = true}
      %t1 = aiex.dma_configure_task_for @outC1 {
        aie.dma_bd(%out : memref<64xi32> offset = 32 len = 32)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t0)
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t0)
      // The completion token for col 1 is waited on in the col 1 cert job
      // aiex.dma_await_task(%t1)
    }

    // Column 0 CERT job
    aiex.cert.attach_to_group(0) {
      aiex.cert.job(0) {
        // This barrier will be reached after the implicit configuration job has
        // completed, because they will be on different pages
        aiex.cert.remote_barrier (rb1, 0x5)
        // col 0 row 3 lock 5 - release the core
        %addr_c0 = arith.constant 0x7AE050 : i32
        %val_c0 = arith.constant 0x2 : i32
        aiex.npu.write32(%addr_c0, %val_c0) : i32, i32
        aiex.cert.remote_barrier (rb2, 0x5)
      }
    }
  }
}

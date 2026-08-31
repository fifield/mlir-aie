//===- cert_rel_acq_sync.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Two-microcontroller global barrier built from aiex.cert.rel_acq_sync instead
// of aiex.cert.remote_barrier (compare npu3_cert_multi_uc, which is otherwise
// structurally identical). uC0 (column 0) and uC2 (column 1) rendezvous twice
// around the lock writes that release each column's core.
//
// Operand encoding (verified on hardware, npu3 / CERT FW 1.5.0.39).
// REL_ACQ_SYNC takes two *base-less* AIE-array addresses; firmware ORs in
// AXIMM_BASE (0x80000000). It reads rel_address once (a lock release, which
// never blocks), then busy-waits on acq_address until the read returns 1.
//
//   addr(lock, col, is_acq, val) =
//       (col << 25) | LOCK_REQUEST_A | (lock << 10) | (is_acq << 9) | ((val & 0x7f) << 2)
//
// with LOCK_REQUEST_A = 0x14000 for the shim NOC module (aie-rt
// Aie4ShimNocLockMod: LockIdOff 0x400, RelAcqOff 0x200, LockValOff 0x4).
// A positive acquire value is acq_eq (wait until state == val, non-consuming),
// which is what lets every participant be granted from the same lock state.
//
// Barrier A, shim NOC lock 8 of column 0:
//     rel(+1)   = 0x16004      acq_eq(2) = 0x16208
// Barrier B, shim NOC lock 9 of column 0:
//     rel(+1)   = 0x16404      acq_eq(2) = 0x16608
//
// Locks 8/9 are clear of the objectfifo shim locks (0/1, which device
// configuration zeroes) and of the locks CERT reserves for its own remote
// barriers (15 downward), so nothing races their initial value of 0.
//
// IMPORTANT scheduling constraint: unlike REMOTE_BARRIER, REL_ACQ_SYNC
// busy-waits *without yielding* to other jobs on the same uC. A uC parked in
// rel_acq_sync therefore cannot run any of its other jobs. So a rel_acq_sync
// barrier must not be placed anywhere a peer can only arrive by way of work
// that this uC still has to perform. Concretely, uC2's wait_tcts for the
// column-1 DMA completion token depends on the runtime-sequence job that runs
// on uC0; it is therefore sequenced *after* barrier B rather than before it.
// Putting the wait_tcts ahead of barrier B deadlocks the pair.
//
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

    // Core-release locks, init 0; the CERT jobs write 2 to release each core.
    // Lock 5 avoids the objectfifo-generated compute-tile locks.
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
        // Barrier A: without it, column 0's configuration of column 1's locks
        // would race this write32.
        aiex.cert.rel_acq_sync(0x16004, 0x16208)
        // col 1 row 2 lock 5 - release the core
        %addr_c1 = arith.constant 0x26AE050 : i32
        %val_c1 = arith.constant 0x2 : i32
        aiex.npu.write32(%addr_c1, %val_c1) : i32, i32
        // Barrier B. Must precede the wait_tcts: see the note in the header.
        aiex.cert.rel_acq_sync(0x16404, 0x16608)
        aiex.cert.wait_tcts(32, 0, 1)
        aiex.cert.remote_barrier (2, 0x5)
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
        aiex.cert.rel_acq_sync(0x16004, 0x16208)
        // col 0 row 3 lock 5 - release the core
        %addr_c0 = arith.constant 0x7AE050 : i32
        %val_c0 = arith.constant 0x2 : i32
        aiex.npu.write32(%addr_c0, %val_c0) : i32, i32
        aiex.cert.rel_acq_sync(0x16404, 0x16608)
        aiex.cert.remote_barrier (2, 0x5)
      }
    }
  }
}

//===- cert_reg_snapshot.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A small pointer-walking computation performed entirely by CERT control code.
// Job-runner registers carry both addresses and values; SAVE_REGISTER records
// the two computed memory values in the uC debug buffer for the host to
// cross-check against ordinary shim-DMA readback.
//
//   obuf[0] = 0x100                         seed
//   obuf[1] = *(&obuf[0]) + 0x23 = 0x123   dynamic address + dynamic value
//   obuf[2] = obuf[1] + 0x100 = 0x223      constant address + dynamic value
//
// Debug records are `{201, 0x123}` and `{202, 0x223}`.
//
//===----------------------------------------------------------------------===//
module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %core = aie.tile(0, 2)

    aie.packet_flow(0) {
      aie.packet_source<%shim, Control32 : 0>
      aie.packet_dest<%shim, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}

    // tile(0,2) base 0x600000 + buffer offset 0x400.
    %obuf = aie.buffer(%core) {sym_name = "obuf", address = 1024 : i32} : memref<16xi32>
    %lock_go = aie.lock(%core, 0) {init = 0 : i32, sym_name = "lock_go"}
    %lock_done = aie.lock(%core, 1) {init = 0 : i32, sym_name = "lock_done"}

    %one = arith.constant 1 : i32
    aie.mem(%core) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      aie.use_lock(%lock_go, AcquireGreaterEqual, %one)
      aie.dma_bd(%obuf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%lock_done, Release, %one)
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }

    aie.flow(%core, DMA : 0, %shim, DMA : 0)
    aie.shim_dma_allocation @out (%shim, S2MM, 0)

    aiex.cert.job(7) {
      aiex.cert.write32(0x600400, 0x100)

      // r0 walks the output buffer; r1 carries the first computed value.
      aiex.cert.mov(0, 0x600400)
      aiex.cert.read32_d(0, 1)
      aiex.cert.add(1, 0x23)
      aiex.cert.add(0, 4)
      aiex.cert.write32_d(0, 1) {address_is_reg, value_is_reg}
      aiex.cert.save_register(0x600404, 201)

      // Read the intermediate normally and write the second result through a
      // register-held value. MASK_POLL_32 is the completion/liveness guard.
      aiex.cert.read32(2, 0x600404)
      aiex.cert.add(2, 0x100)
      aiex.cert.write32_d(0x600408, 2) {value_is_reg}
      aiex.cert.maskpoll32(0x600408, 0x00000fff, 0x223)
      aiex.cert.save_register(0x600408, 202)

      // Release the DMA that returns obuf to the host.
      aiex.cert.write32(0x6ae000, 1)
    }

    aie.runtime_sequence @sequence(%out: memref<16xi32>) {
      %task = aiex.dma_configure_task_for @out {
        aie.dma_bd(%out : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%task)
      aiex.dma_await_task(%task)
    }
  }
}

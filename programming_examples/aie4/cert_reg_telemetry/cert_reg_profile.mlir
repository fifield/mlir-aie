//===- cert_reg_profile.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SAVE_TIMESTAMPS turns the uC debug buffer into a lightweight profiler. This
// example measures a register-only arithmetic block and an AXI-MM read/modify/
// write block in one dispatch. The host reports both gaps after checking the
// record order and the independently DMA-read computation results.
//
//   101 --[ 32 x ADD ]-- 102 --[ baseline ]-- 103 --[ 8 x MMIO RMW ]-- 104
//
// The result buffer proves both measured regions executed: r0 becomes 32 and
// obuf[0] is incremented from 1 to 9.
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
      aiex.cert.write32(0x600400, 1)
      aiex.cert.mov(0, 0)

      aiex.cert.save_timestamps(101)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      aiex.cert.save_timestamps(102)

      // Adjacent stamps provide the record/write overhead baseline.
      aiex.cert.save_timestamps(103)

      aiex.cert.mov(1, 0x600400)
      aiex.cert.read32_d(1, 2)
      aiex.cert.add(2, 1)
      aiex.cert.write32_d(1, 2) {address_is_reg, value_is_reg}
      aiex.cert.read32_d(1, 2)
      aiex.cert.add(2, 1)
      aiex.cert.write32_d(1, 2) {address_is_reg, value_is_reg}
      aiex.cert.read32_d(1, 2)
      aiex.cert.add(2, 1)
      aiex.cert.write32_d(1, 2) {address_is_reg, value_is_reg}
      aiex.cert.read32_d(1, 2)
      aiex.cert.add(2, 1)
      aiex.cert.write32_d(1, 2) {address_is_reg, value_is_reg}
      aiex.cert.read32_d(1, 2)
      aiex.cert.add(2, 1)
      aiex.cert.write32_d(1, 2) {address_is_reg, value_is_reg}
      aiex.cert.read32_d(1, 2)
      aiex.cert.add(2, 1)
      aiex.cert.write32_d(1, 2) {address_is_reg, value_is_reg}
      aiex.cert.read32_d(1, 2)
      aiex.cert.add(2, 1)
      aiex.cert.write32_d(1, 2) {address_is_reg, value_is_reg}
      aiex.cert.read32_d(1, 2)
      aiex.cert.add(2, 1)
      aiex.cert.write32_d(1, 2) {address_is_reg, value_is_reg}
      aiex.cert.save_timestamps(104)

      aiex.cert.write32_d(0x600404, 0) {value_is_reg}
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

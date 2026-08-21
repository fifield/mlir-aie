//===- cert_save_register.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `aiex.cert.save_register` on real npu3 hardware.
//
// SAVE_REGISTER reads a 32-bit AIE-array location and appends an 8-byte record
// `{unq_id, value}` to the microcontroller's local debug staging buffer, which
// firmware flushes to a host DDR buffer at the end of every task
// (`copy_dm2mm_for_timestamp(FORCE_FLUSH)`, cert `dpu/dpu_control.c:2374`).
// The host gets at that buffer through XRT's internal `bo_int` API -- see
// test.cpp. A user-facing application of the same telemetry mechanism lives in
// programming_examples/aie4/cert_reg_telemetry.
//
// Coreless design (no aie.core): a single device-level `aiex.cert.job` writes
// known words into tile(0,2) data memory and into tile control space,
// SAVE_REGISTERs each one, then releases a gate lock that lets a plain
// core-DMA -> shim-S2MM transfer stream that same memory back to the host.
// So there are two independent views of the same locations -- the SAVE_REGISTER
// records and the DMA'd buffer -- and test.cpp cross-checks them against each
// other.
//
// The point of writing 0x600400 twice is to prove SAVE_REGISTER samples the
// *live* contents at the instant it executes:
//
//   record 201 == 0x11112222   (0x600400 before the overwrite)
//   record 203 == 0x55556666   (0x600400 after  the overwrite)  == obuf[0]
//
// 0x11112222 is nowhere in the array by the time the DMA runs, so a
// SAVE_REGISTER that returned a compile-time immediate, a cached value, or the
// final memory contents could not produce both records.
//
// Addresses are the *base-less* AIE-array form; firmware ORs in AXIMM_BASE
// (0x80000000) itself -- `control_op_save_register` does
// `reg_read32((void *)(AXIMM_BASE | address))` (`dpu/dpu_control.c:1471`), the
// same convention as cert.write32 / cert.read32.
//
//   addr = (col << 25) | (addrRow << 20) | offset,  addrRow = row + 5*memRows - 1
//
// npu3 has one mem-tile row, so tile(0,2) -> addrRow 6 -> tile base 0x600000.
//   obuf (buffer offset 0x400)      -> 0x600400, obuf[i] at 0x600400 + 4i
//   core-tile lock N value register -> 0x600000 | 0xAE000 | (0x10 * N)
//     lock 0 (gate) -> 0x6AE000, lock 1 (done) -> 0x6AE010
//
//===----------------------------------------------------------------------===//
module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %core = aie.tile(0, 2)

    aie.packet_flow(0) {
      aie.packet_source<%shim, Control32 : 0>
      aie.packet_dest<%shim, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}

    // obuf at tile-local offset 0x400 -> absolute 0x600400.
    %obuf = aie.buffer(%core) {sym_name = "obuf", address = 1024 : i32} : memref<16xi32>

    // Gate lock: released by the cert job once it is done poking obuf.
    %lock_go = aie.lock(%core, 0) {init = 0 : i32, sym_name = "lock_go"}
    %lock_done = aie.lock(%core, 1) {init = 0 : i32, sym_name = "lock_done"}

    %c1_lock = arith.constant 1 : i32
    aie.mem(%core) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      aie.use_lock(%lock_go, AcquireGreaterEqual, %c1_lock)
      aie.dma_bd(%obuf : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%lock_done, Release, %c1_lock)
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }

    aie.flow(%core, DMA : 0, %shim, DMA : 0)
    aie.shim_dma_allocation @out (%shim, S2MM, 0)

    aiex.cert.job(7) {
      // --- record 201: a word in core-tile data memory ----------------------
      // The poll is a liveness check on the write, not on SAVE_REGISTER: if
      // the write never landed the job hangs here instead of quietly saving a
      // zero, which would be indistinguishable from a broken SAVE_REGISTER.
      aiex.cert.write32(0x600400, 0x11112222)
      aiex.cert.poll32(0x600400, 0x11112222)
      aiex.cert.save_register(0x600400, 201)

      // --- record 202: a different word, a different value ------------------
      // A SAVE_REGISTER that ignored its address operand and always read
      // 0x600400 would produce 0x11112222 here.
      aiex.cert.write32(0x600404, 0x33334444)
      aiex.cert.poll32(0x600404, 0x33334444)
      aiex.cert.save_register(0x600404, 202)

      // --- record 203: 0x600400 again, after overwriting it -----------------
      // Same address as record 201, different contents. This is the assertion
      // that SAVE_REGISTER reads memory at execution time.
      aiex.cert.write32(0x600400, 0x55556666)
      aiex.cert.poll32(0x600400, 0x55556666)
      aiex.cert.save_register(0x600400, 203)

      // --- record 204: tile CONTROL space, not data memory -------------------
      // Lock 1's value register. 21 is not a value the lock can reach on its
      // own from init 0 before the DMA runs, so a stale/zero read is visible.
      aiex.cert.write32(0x6AE010, 21)
      aiex.cert.poll32(0x6AE010, 21)
      aiex.cert.save_register(0x6AE010, 204)

      // --- release the gate so the MM2S BD streams obuf to the host ---------
      // A plain WRITE_32 on purpose: the gate is this test's liveness
      // mechanism, so it must not depend on the op under test.
      aiex.cert.write32(0x6AE000, 1)
    }

    aie.runtime_sequence @sequence(%out: memref<16xi32>) {
      %t = aiex.dma_configure_task_for @out {
        aie.dma_bd(%out : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

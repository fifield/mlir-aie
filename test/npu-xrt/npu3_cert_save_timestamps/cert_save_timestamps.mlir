//===- cert_save_timestamps.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `aiex.cert.save_timestamps` on real npu3 hardware. Also exercises
// `aiex.cert.sleep`, used below as the workload that produces a measurable
// gap between two timestamps -- see the ratio assertion further down.
//
// SAVE_TIMESTAMPS reads the microcontroller's 64-bit tick counter and appends
// a 12-byte record `{unq_id, ts_high, ts_low}` to the uC's local debug staging
// buffer, which firmware flushes to a host DDR buffer at the end of every task
// (`copy_dm2mm_for_timestamp(FORCE_FLUSH)`, cert `dpu/dpu_control.c:2374`).
// The host gets at that buffer through XRT's internal `bo_int` API -- see
// test.cpp. A user-facing application of the same telemetry mechanism lives in
// programming_examples/aie4/cert_reg_telemetry.
//
// Records carry no type tag, so this design deliberately contains no
// `cert.save_register`: mixing the two op families in one job would interleave
// 12-byte and 8-byte records in one untagged buffer that no reader can decode.
// `cert.save_register` has its own test in ../npu3_cert_save_register.
//
// The four stamps bracket one `cert.sleep(1000)`, i.e. a 1000 us busy-wait
// (`control_op_sleep`, `dpu/dpu_control.c:1116`, target is in microseconds):
//
//   101 --------- 102 ===[ sleep 1000us ]=== 103 --------- 104
//        gap A               gap B                gap C
//
// so gap B must dwarf gaps A and C, which are back-to-back opcodes. test.cpp
// asserts that as a ratio, never as an absolute cycle count, because the tick
// counter is a free-running clock and the run-to-run values vary.
//
// Coreless design (no aie.core): the cert job stamps, drops a marker word into
// tile(0,2) data memory, then releases a gate lock that lets a plain core-DMA
// -> shim-S2MM transfer stream that memory back to the host. The marker is
// this test's liveness check -- it proves the job ran to completion even if
// the debug buffer readback path is broken, so "no records" and "job never
// ran" are distinguishable failures.
//
// Addresses are the *base-less* AIE-array form; firmware ORs in AXIMM_BASE
// (0x80000000) itself.
//
//   addr = (col << 25) | (addrRow << 20) | offset,  addrRow = row + 5*memRows - 1
//
// npu3 has one mem-tile row, so tile(0,2) -> addrRow 6 -> tile base 0x600000.
//   obuf (buffer offset 0x400)      -> 0x600400, obuf[i] at 0x600400 + 4i
//   core-tile lock N value register -> 0x600000 | 0xAE000 | (0x10 * N)
//     lock 0 (gate) -> 0x6AE000
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

    // Gate lock: released by the cert job once it is done stamping.
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
      // --- gap A: two adjacent stamps, no work in between -------------------
      // Whatever this costs is the floor on what SAVE_TIMESTAMPS itself costs;
      // it is the baseline gap B is compared against.
      aiex.cert.save_timestamps(101)
      aiex.cert.save_timestamps(102)

      // --- gap B: the same two stamps with 1000 us of busy-wait between ------
      aiex.cert.sleep(1000)
      aiex.cert.save_timestamps(103)

      // --- gap C: a second no-work baseline, after the sleep -----------------
      // Having a baseline on both sides rules out "the gap grew because
      // everything after the first stamp is slower", and gives test.cpp two
      // independent baselines to compare gap B against.
      aiex.cert.save_timestamps(104)

      // --- liveness marker: distinguishes "job never ran" from "no records" --
      aiex.cert.write32(0x600400, 0xA5A5A5A5)
      aiex.cert.poll32(0x600400, 0xA5A5A5A5)

      // --- release the gate so the MM2S BD streams obuf to the host ---------
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

//===- cert_read_write32d.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tier-1 cert ISA register/scalar ops on real npu3 hardware: cert.read32,
// cert.read32_d, cert.poll32, cert.maskpoll32, plus uC WRITE_32 and WRITE_32_D
// straight into core-tile *data* memory and into tile control space.
//
// Coreless design (no aie.core): a single device-level `aiex.cert.job` writes
// and inspects tile(0,2) data memory, then releases a gate lock that lets a
// plain core-DMA -> shim-S2MM transfer stream that memory back to the host, so
// every claim below is checked byte-for-byte by test.cpp.
//
// The cert job is written *before* `aie.runtime_sequence`, which puts it on the
// same page as (and ahead of) the runtime-sequence job, one page after the
// implicit device-configuration job:
//
//   page 1: START_JOB 1   device configuration (lock init, DMA BDs, routing)
//   .eop
//   page 2: START_JOB 7   this job
//           START_JOB 8   runtime sequence (shim BD patch + start + WAIT_TCTS)
//   .eop
//
// Pages on a uC run strictly in order, so the lock inits from page 1 cannot
// race the lock release here, and jobs inside a page start in textual order,
// so job 7 runs to completion before job 8 configures the shim DMA.
//
// Address convention (verified on hardware, npu3 / CERT FW 1.5.0.39):
// read32 / read32_d / poll32 / maskpoll32 / write32 / write32_d all take the
// *base-less* AIE-array address, exactly like the WRITE_32 lock pokes in
// npu3_cert_multi_uc -- firmware ORs in AXIMM_BASE (0x80000000) itself.
// Passing 0x80000000|addr explicitly also works (bit 31 is idempotent), but
// base-less is the canonical form used throughout this repo.
//
//   addr = (col << 25) | (addrRow << 20) | offset,  addrRow = row + 5*memRows - 1
//
// npu3 has one mem-tile row, so tile(0,2) -> addrRow 6 -> tile base 0x600000.
//   obuf (buffer offset 0x400)             -> 0x600400, obuf[i] at 0x600400+4i
//   core-tile lock N value register        -> 0x600000 | 0xAE000 | (0x10 * N)
//     lock 0 (gate) -> 0x6AE000, lock 1 (done) -> 0x6AE010
//
// WHAT THIS PROVES
//
// cert.write32_d used to wedge the microcontroller in all four flags
// encodings. That was a COMPILER bug -- the ISA flags bits mean is-REGISTER
// and the emitter emitted their bitwise inverse, so firmware read a large
// immediate as a register index, got CONTROL_INVALID_REG_VAL (0xDEADBEEF)
// back, and issued a store to an unmapped AXI-MM address that never
// completed. That is fixed (lib/Targets/AIETargetUcCert.cpp; the four
// encodings are pinned at assembly level by
// test/Targets/AIETargetUcCert/cert_write32_d_flags.mlir and this exact op
// sequence by test/Targets/AIETargetUcCert/cert_reg_roundtrip.mlir).
//
// Because WRITE_32_D works, a cert register's *contents* are now
// host-observable, so this test no longer stops at "the register ops execute
// without faulting" -- it asserts the values:
//
//   obuf[0] = 0x11112222  WRITE_32 immediate
//   obuf[1] = 0x33334444  WRITE_32 immediate
//   obuf[2] = 0x11112222  READ_32 of obuf[0]   -> $r0 -> WRITE_32_D flags 2
//   obuf[3] = 0x33334444  READ_32_D of obuf[1] -> $r2 -> WRITE_32_D flags 2
//   obuf[4] = 0x00600404  $r1 at $r3                 -> WRITE_32_D flags 3
//   obuf[5] = 0x55556666  immediate at $r4           -> WRITE_32_D flags 1
//   obuf[6] = 0x77778888  immediate at immediate     -> WRITE_32_D flags 0
//   obuf[7] = 0x00000015  READ_32 of a lock value register that WRITE_32_D
//                         (flags 1) had just set to 21 -> WRITE_32_D flags 2
//   obuf[8..11]           pointer table, WRITE_32 immediates (see below)
//
// The ISA has no way to put an immediate into a register (MOV and ADD were
// removed in aiebu #397), so every register-held address is first stored in
// the obuf[8..11] pointer table with WRITE_32 and then loaded with READ_32:
//
//   obuf[8]  = 0x00600404  -> $r1, address of obuf[1]
//   obuf[9]  = 0x00600410  -> $r3, address of obuf[4]
//   obuf[10] = 0x00600414  -> $r4, address of obuf[5]
//   obuf[11] = 0x006AE010  -> $r5, lock 1 value register
//
// obuf[2] and obuf[3] can only hold those words if READ_32 / READ_32_D really
// returned the data at the addresses they were given. Each register above
// holds a distinct value, so obuf[4] also catches WRITE_32_D resolving the
// wrong register for either operand. All four WRITE_32_D flags encodings
// appear above, so each one gets real hardware coverage and not just a
// FileCheck line.
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
      // --- obuf[0]: uC WRITE_32 into core-tile data memory ------------------
      aiex.cert.write32(0x600400, 0x11112222)
      // The uC read path must see that write at the same base-less address.
      // A wrong address decode never matches and the run times out.
      aiex.cert.maskpoll32(0x600400, 0xFFFFFFFF, 0x11112222)
      // Same word, upper half only, through the masked poll.
      aiex.cert.maskpoll32(0x600400, 0xFFFF0000, 0x11110000)

      // --- obuf[1]: second immediate, so a stuck/echoed value is visible ----
      aiex.cert.write32(0x600404, 0x33334444)
      aiex.cert.maskpoll32(0x600404, 0xFFFFFFFF, 0x33334444)

      // --- obuf[8..11]: pointer table for the register-held addresses -------
      aiex.cert.write32(0x600420, 0x600404)
      aiex.cert.write32(0x600424, 0x600410)
      aiex.cert.write32(0x600428, 0x600414)
      aiex.cert.write32(0x60042C, 0x6AE010)

      // --- obuf[2]: READ_32 round-trip --------------------------------------
      // READ_32: constant address -> destination register (dest operand
      // first). Echo $r0 to obuf[2] with WRITE_32_D {value_is_reg} (flags 2):
      // immediate address, value taken from a register. If READ_32 returned
      // anything but the word at 0x600400, obuf[2] shows it.
      aiex.cert.read32(0, 0x600400)
      aiex.cert.write32_d(0x600408, 0) {value_is_reg}

      // --- obuf[3]: READ_32_D round-trip ------------------------------------
      // READ_32 the address of obuf[1] from the pointer table into $r1, then
      // READ_32_D: address register first, destination register second. Echo
      // $r2 to obuf[3], again with flags 2.
      aiex.cert.read32(1, 0x600420)
      aiex.cert.read32_d(1, 2)
      aiex.cert.write32_d(0x60040C, 2) {value_is_reg}

      // --- obuf[4]: WRITE_32_D {address_is_reg, value_is_reg} (flags 3) ----
      // Both operands come from registers: store $r1 (the obuf[1] pointer,
      // 0x00600404) at the address loaded into $r3.
      aiex.cert.read32(3, 0x600424)
      aiex.cert.write32_d(3, 1) {address_is_reg, value_is_reg}

      // --- obuf[5]: WRITE_32_D {address_is_reg} (flags 1) -------------------
      // Register-held address, immediate value.
      aiex.cert.read32(4, 0x600428)
      aiex.cert.write32_d(4, 0x55556666) {address_is_reg}

      // --- obuf[6]: WRITE_32_D with no flags (flags 0) ----------------------
      // Both operands immediate; the WRITE_32 equivalent, through the dynamic
      // opcode. This is the encoding the old inverted emitter turned into
      // "both operands are registers", which is what wedged the uC.
      aiex.cert.write32_d(0x600418, 0x77778888)

      // --- obuf[7]: WRITE_32_D into tile CONTROL space, read back, echo -----
      // Drive an unrelated lock's value register with plain WRITE_32 and read
      // it straight back (same round-trip as obuf[0], but against tile control
      // space instead of data memory)...
      aiex.cert.write32(0x6AE010, 5)
      aiex.cert.maskpoll32(0x6AE010, 0xFFFFFFFF, 5)
      // ...then overwrite it with WRITE_32_D through a register-held address
      // and READ_32 the result out to obuf[7]. 21 != 5, so a WRITE_32_D that
      // silently did nothing here leaves 5 in obuf[7] and fails loudly rather
      // than looking untouched.
      aiex.cert.read32(5, 0x60042C)
      aiex.cert.write32_d(5, 21) {address_is_reg}
      aiex.cert.read32(6, 0x6AE010)
      aiex.cert.write32_d(0x60041C, 6) {value_is_reg}

      // --- release the gate so the MM2S BD streams obuf to the host ---------
      // Kept as a plain WRITE_32 on purpose: the gate is the test's liveness
      // mechanism, so it must not depend on the op under test. A broken
      // WRITE_32_D then shows up as wrong words in obuf (a diagnosable FAIL
      // with got/expected) instead of a bare 10 s timeout.
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

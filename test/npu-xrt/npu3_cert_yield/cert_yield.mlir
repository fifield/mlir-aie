//===- cert_yield.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tier-1 cert ISA `cert.yield` (ISA YIELD, opcode 0x08) on real npu3 hardware.
//
// YIELD takes no operands and produces no value, so the only way to observe it
// is to build a job that *cannot* produce the right answer unless the yield
// really performs a context switch to another job and control really comes
// back afterwards. That is what this design is.
//
// THE MECHANISM
//
// Two device-level cert jobs land on the same page, in textual order, ahead of
// the auto-generated runtime-sequence job:
//
//   job A (job 7)  poison a word, YIELD, then sample that same word
//   job B (job 8)  overwrite the word with the real value, then end
//   job C (job 9)  runtime sequence: shim BD patch + start + WAIT_TCTS
//
// Job A samples the word with `cert.read32` on the instruction *immediately*
// after the YIELD -- no `cert.poll32`, no other blocking op in between -- so a
// YIELD that did nothing would be directly visible as A reading back its own
// poison. Only explicit yield/poll/wait ops cede control in this ISA; plain
// write32/read32/mov/add do not. So without the YIELD, job A runs start to
// finish in its own slot and job B never gets to run before A's sample.
//
//   obuf[0] == 0xB1B1B1B1  ->  YIELD switched to job B, B ran to completion,
//                              and control came back to A
//   obuf[0] == 0x0BAD0001  ->  YIELD was a no-op
//
// That second case is a measured negative control, not an inference. On npu3,
// deleting the single `aiex.cert.yield` line below and rerunning this exact
// design makes obuf[0] come back as 0x0BAD0001 instead of 0xB1B1B1B1 (3/3
// runs), and obuf[2]/obuf[3] come back pre-job-B as well, because job A
// releases the gate as its last instruction and the core MM2S BD latches all
// 16 words into the stream before job B is ever scheduled. Nothing hangs; the
// failure is three got/expected lines. A separate three-job probe (an extra
// job inserted textually between A and B) showed that one YIELD gives *every*
// other runnable job on the page a turn before A is re-scheduled, so this
// design is robust to either scheduling rule.
//
// obuf[4] is the same word sampled *before* the YIELD; it must be the poison,
// which is what makes obuf[0] meaningful rather than a lucky initial value.
//
// Coreless design (no aie.core), the same shape as npu3_cert_read_write32d and
// npu3_cert_ucdma: cert jobs poke tile(0,2) data memory, then a plain
// core-DMA -> shim-S2MM transfer streams that memory back to the host, gated on
// a lock that job A releases with a plain `cert.write32`. The gate release is
// never the op under test, so a broken YIELD shows up as a wrong word with a
// got/expected line rather than as a bare timeout.
//
// Addresses are the *base-less* AIE-array form; firmware ORs in AXIMM_BASE
// (0x80000000) itself.
//
//   addr = (col << 25) | (addrRow << 20) | offset,  addrRow = row + 5*memRows - 1
//
// npu3 has one mem-tile row, so tile(0,2) -> addrRow 6 -> tile base 0x600000.
//   obuf (buffer offset 0x400)      -> 0x600400, obuf[i] at 0x600400 + 4i
//   core-tile lock N value register -> 0x600000 | 0xAE000 | (0x10 * N)
//     lock 0 (gate) -> 0x6AE000, lock 1 (done) -> 0x6AE010
//
// Buffer map (obuf, 16 words):
//   obuf[0]     0x600400  A's sample of T taken right AFTER the yield
//   obuf[1]     0x600404  "job A ran" marker, written before the yield
//   obuf[2]     0x600408  "job B ran" marker, written by job B
//   obuf[3]     0x60040C  T itself: poisoned by A, overwritten by B
//   obuf[4]     0x600410  A's sample of T taken BEFORE the yield (the poison)
//   obuf[5..15]           untouched -> 0
//
// T deliberately lives *inside* obuf, so the host sees both what job A read and
// what the word actually ended up holding.
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

    // Gate lock: released by job A once it has sampled the rendezvous word.
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

    // ======================= job A: the yielding job ========================
    // Written first, so it is emitted first on the page and therefore runs
    // first (job_id is a label, not an order -- see
    // test/Targets/AIETargetUcCert/job_textual_order.mlir).
    aiex.cert.job(7) {
      // T = obuf[3]. Poison it, so "job B never ran" is a distinct value from
      // the host sentinel (0xDEADBEEF) and from untouched memory (0).
      aiex.cert.write32(0x60040C, 0x0BAD0001)

      // Sample T *before* the yield -> obuf[4]. This must come back as the
      // poison; it is the control that proves T really held the poison at the
      // moment the yield executed, so obuf[0] != poison means something
      // changed T in between, and the only thing that can is job B.
      aiex.cert.read32(0, 0x60040C)
      aiex.cert.write32_d(0x600410, 0) {value_is_reg}

      // "job A reached the yield" marker -> obuf[1].
      aiex.cert.write32(0x600404, 0xA1A1A1A1)

      // ---- the op under test ---------------------------------------------
      // Context switch to the next available job on this page, which is job B.
      // Job B has no blocking ops, so it runs straight through and ends, and
      // job A is re-scheduled in the next scheduling cycle.
      aiex.cert.yield

      // ---- sample T IMMEDIATELY after resuming -----------------------------
      // No poll, no wait, nothing in between: this instruction is the whole
      // experiment. If YIELD did not cede control, $r1 holds 0x0BAD0001.
      aiex.cert.read32(1, 0x60040C)
      aiex.cert.write32_d(0x600400, 1) {value_is_reg}

      // ---- release the gate so the MM2S BD streams obuf to the host -------
      // Plain WRITE_32 on purpose: the gate must never depend on the op under
      // test, so a broken YIELD gives diagnosable words instead of a hang.
      aiex.cert.write32(0x6AE000, 1)
    }

    // ======================= job B: the yield target ========================
    // Written second, so it is the next job in the page's rotation after A.
    // No blocking ops, no explicit terminator: the region just ends and the
    // emitter closes it with END_JOB.
    aiex.cert.job(8) {
      // "job B ran" marker -> obuf[2]. Written before T, so obuf[2] being set
      // while obuf[0] still holds poison would mean B ran but only *after* A
      // had already resumed.
      aiex.cert.write32(0x600408, 0xB2B2B2B2)

      // The real value, into the rendezvous word T = obuf[3]. Distinct from
      // the poison, from the sentinel, from 0, and from both markers.
      aiex.cert.write32(0x60040C, 0xB1B1B1B1)
    }

    // job C is generated from this runtime sequence and lands third on the
    // same page (START_JOB 9).
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

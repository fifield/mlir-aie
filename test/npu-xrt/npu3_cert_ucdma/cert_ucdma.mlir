//===- cert_ucdma.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tier-1 cert ISA uC-DMA ops on real npu3 hardware: the async pair
// cert.uc_dma_write_des + cert.wait_uc_dma, and the blocking
// cert.uc_dma_write_des_sync, driving cert.uc_dma_chain / cert.uc_dma_bd
// descriptor chains whose payloads are `memref.global` constants embedded in
// the control page.
//
// Coreless design (no aie.core), same shape as npu3_cert_read_write32d: one
// device-level `aiex.cert.job` runs the uC-DMA transfers into tile(0,2) data
// memory and then releases a gate lock that lets a plain core-DMA -> shim-S2MM
// transfer stream that memory back to the host. The uC-DMA destinations are
// *inside* the buffer that gets DMAed out, so the bytes the uC-DMA moved are
// literally the bytes test.cpp asserts on -- not an echo of them.
//
// NAMING GOTCHA (the reason this test exists)
//
// On `cert.uc_dma_bd`, MLIR's operand names are the inverse of what the ISA
// spec's prose calls them:
//
//   remote_address = FlatSymbolRefAttr naming a device-level
//                    `memref.global "private" constant` -- the SOURCE data,
//                    which the emitter inlines as `.long`s into the control
//                    page (lib/Targets/AIETargetUcCert.cpp, emitUcDmaBdData).
//   local_address  = the DESTINATION AIE-array address the uC-DMA writes to.
//
// So `aiex.cert.uc_dma_bd @sym, 0x600500, 256, true` reads 256 words from the
// control-page copy of @sym and writes them to AIE-array address 0x600500.
// "local"/"remote" are from the AIE array's point of view, not the uC's.
//
// `next_bd` is "there is another BD after this one", so a chain ends with
// false. The 2-BD @chain_async below covers that polarity on hardware: a
// flipped next_bd leaves BD1's destination block poisoned.
//
// Address convention (same as npu3_cert_read_write32d): base-less AIE-array
// addresses, addr = (col << 25) | (addrRow << 20) | offset, addrRow = row +
// 5*memRows - 1. npu3 has one mem-tile row, so tile(0,2) -> addrRow 6 -> tile
// base 0x600000.
//
//   obuf (buffer offset 0x400)        -> 0x600400, obuf[i] at 0x600400 + 4i
//   core-tile lock N value register   -> 0x6AE000 + 0x10*N
//
// Buffer map (obuf, 512 words):
//   obuf[0]        0x600400  echo of the LAST word of @chain_async BD0
//   obuf[1]        0x600404  echo of the LAST word of @chain_async BD1
//   obuf[2]        0x600408  echo of the LAST word of @chain_sync's BD
//   obuf[3..63]              untouched -> 0
//   obuf[64..319]  0x600500  @chain_async BD0 destination, 256 words
//   obuf[320..335] 0x600900  @chain_async BD1 destination, 16 words
//   obuf[336..383]           untouched -> 0
//   obuf[384..399] 0x600A00  @chain_sync destination, 16 words
//   obuf[400..511]           untouched -> 0
//
// WHY THE ECHOES ARE READ *IMMEDIATELY* AFTER THE WAIT
//
// The three cert.read32 + cert.write32_d {value_is_reg} echoes sample the last
// word of each destination range with no intervening cert.poll32. That
// ordering is the only thing that could make a no-op cert.wait_uc_dma
// observable: if the wait did not actually block, the read would race the
// in-flight transfer and see poison. The cert.poll32s that follow are a
// separate liveness net -- they make a genuinely stuck transfer time out
// cleanly instead of returning a confusing partial buffer.
//
// That is not hypothetical. Measured on npu3: deleting the cert.wait_uc_dma
// line and rerunning this exact design makes obuf[0] come back as 0 instead of
// 0xA5A5A501 (3/3 runs) -- the uC read the tail of BD0's 256-word destination
// while the transfer was still in flight. Every other word, including the two
// later echoes, still comes back correct, because by then enough uC
// instructions have retired for the transfer to have finished. So the wait is
// load-bearing, and this is the word that shows it.
//
// POISON DISCIPLINE
//
// Before each transfer the job pokes a distinct 0x0BAD000N into the first word
// of that transfer's destination with a plain cert.write32. That separates
// "uC-DMA never ran" (poison) from "DMA never covered this word" (host
// sentinel 0xDEADBEEF) from "nothing touched it" (0). The gate release stays a
// plain cert.write32 so a broken op under test fails with diagnosable values
// rather than an opaque hang.
//
//===----------------------------------------------------------------------===//
module {
  aie.device(npu3) {
    // ---- uC-DMA payloads: constants embedded in the control page -----------
    // The emitter walks these symbols from the cert.uc_dma_bd remote_address
    // and writes their words out as `.long`s in the chain's data section.

    // 256-word splat -> @chain_async BD0. Large enough that the transfer is
    // not instantaneous, which is what gives the wait something to wait on.
    memref.global "private" constant @ucdma_big : memref<256xi32> = dense<0xA5A5A501>

    // 16 distinct words -> @chain_async BD1. Distinct values catch a
    // transfer that lands at the wrong offset or repeats a word.
    memref.global "private" constant @ucdma_tail : memref<16xi32> = dense<[
      0x11110001, 0x11110002, 0x11110003, 0x11110004,
      0x11110005, 0x11110006, 0x11110007, 0x11110008,
      0x11110009, 0x1111000A, 0x1111000B, 0x1111000C,
      0x1111000D, 0x1111000E, 0x1111000F, 0x11110010]>

    // 16 distinct words -> @chain_sync, in a disjoint value range so a
    // mixed-up chain is obvious.
    memref.global "private" constant @ucdma_sync : memref<16xi32> = dense<[
      0x22220001, 0x22220002, 0x22220003, 0x22220004,
      0x22220005, 0x22220006, 0x22220007, 0x22220008,
      0x22220009, 0x2222000A, 0x2222000B, 0x2222000C,
      0x2222000D, 0x2222000E, 0x2222000F, 0x22220010]>

    // ---- descriptor chains --------------------------------------------------
    // Two BDs, so next_bd=true (continue) / next_bd=false (end) both get real
    // hardware coverage.
    aiex.cert.uc_dma_chain @chain_async {
      aiex.cert.uc_dma_bd @ucdma_big, 0x600500, 256, true
      aiex.cert.uc_dma_bd @ucdma_tail, 0x600900, 16, false
    }
    aiex.cert.uc_dma_chain @chain_sync {
      aiex.cert.uc_dma_bd @ucdma_sync, 0x600A00, 16, false
    }

    %shim = aie.tile(0, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %core = aie.tile(0, 2)

    aie.packet_flow(0) {
      aie.packet_source<%shim, Control32 : 0>
      aie.packet_dest<%shim, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}

    // obuf at tile-local offset 0x400 -> absolute 0x600400.
    %obuf = aie.buffer(%core) {sym_name = "obuf", address = 1024 : i32} : memref<512xi32>

    // Gate lock: released by the cert job once every transfer has landed.
    %lock_go = aie.lock(%core, 0) {init = 0 : i32, sym_name = "lock_go"}
    %lock_done = aie.lock(%core, 1) {init = 0 : i32, sym_name = "lock_done"}

    %c1_lock = arith.constant 1 : i32
    aie.mem(%core) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      aie.use_lock(%lock_go, AcquireGreaterEqual, %c1_lock)
      aie.dma_bd(%obuf : memref<512xi32> offset = 0 len = 512)
      aie.use_lock(%lock_done, Release, %c1_lock)
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }

    aie.flow(%core, DMA : 0, %shim, DMA : 0)
    aie.shim_dma_allocation @out (%shim, S2MM, 0)

    aiex.cert.job(7) {
      // --- poison every uC-DMA destination ---------------------------------
      // Distinct per destination, so a FAIL says which transfer never ran.
      aiex.cert.write32(0x600500, 0x0BAD0001)   // obuf[64],  @chain_async BD0
      aiex.cert.write32(0x600900, 0x0BAD0002)   // obuf[320], @chain_async BD1
      aiex.cert.write32(0x600A00, 0x0BAD0003)   // obuf[384], @chain_sync

      // --- async: enqueue, then wait ---------------------------------------
      // The wait handle is job-private $r0 (the verifier requires r0..r7 with
      // its reaching definition in the same job).
      aiex.cert.uc_dma_write_des(0, @chain_async)
      aiex.cert.wait_uc_dma(0)

      // --- sample the tail of each block IMMEDIATELY after the wait --------
      // No poll in between on purpose (see header). 0x6008FC is obuf[319],
      // the last word of BD0; 0x60093C is obuf[335], the last word of BD1.
      aiex.cert.read32(1, 0x6008FC)
      aiex.cert.write32_d(0x600400, 1) {value_is_reg}
      aiex.cert.read32(2, 0x60093C)
      aiex.cert.write32_d(0x600404, 2) {value_is_reg}

      // --- liveness net ------------------------------------------------------
      // First word of each async destination. If a transfer is genuinely
      // stuck these never match and the host's bounded wait reports a clean
      // timeout rather than a half-written buffer.
      aiex.cert.poll32(0x600500, 0xA5A5A501)
      aiex.cert.poll32(0x600900, 0x11110001)

      // --- sync: single blocking op ----------------------------------------
      // uc_dma_write_des_sync enqueues and waits in one instruction, so the
      // read that follows must already see the full payload. 0x600A3C is
      // obuf[399], the last word of the sync destination.
      aiex.cert.uc_dma_write_des_sync(@chain_sync)
      aiex.cert.read32(3, 0x600A3C)
      aiex.cert.write32_d(0x600408, 3) {value_is_reg}
      aiex.cert.poll32(0x600A00, 0x22220001)

      // --- release the gate so the MM2S BD streams obuf to the host --------
      // Plain WRITE_32 on purpose: the gate must never depend on an op under
      // test, so a broken uC-DMA op yields wrong words instead of a hang.
      aiex.cert.write32(0x6AE000, 1)
    }

    aie.runtime_sequence @sequence(%out: memref<512xi32>) {
      %t = aiex.dma_configure_task_for @out {
        aie.dma_bd(%out : memref<512xi32> offset = 0 len = 512) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

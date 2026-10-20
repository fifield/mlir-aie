//===- cert_reg_roundtrip.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Offline guard for the cert register round-trip that
// test/npu-xrt/npu3_cert_read_write32d runs on npu3: read a known word into a
// register with READ_32 / READ_32_D and carry it back into the array with
// WRITE_32_D so the host can see it. Register-held addresses are loaded with
// READ_32 from a pointer table, since the ISA no longer has MOV/ADD.
//
// cert_write32_d_flags.mlir pins the four WRITE_32_D flags encodings in
// isolation; this pins them in the exact op sequence and operand order the
// hardware tests use, which is where a polarity or operand-order regression
// actually bites. WRITE_32_D's flags bits are is-REGISTER
// (flags = (address_is_reg?1:0) | (value_is_reg?2:0)); emitting the inverse
// makes firmware treat an immediate as a register index and wedges the
// microcontroller, so a silent regression here costs a hardware timeout rather
// than a wrong answer.
//
// Note the operand orders, which differ between the two read ops and are easy
// to get backwards (both spellings parse):
//   cert.read32(<dest reg>, <address>)      -- destination register first
//   cert.read32_d(<address reg>, <dest reg>) -- address register first

// RUN: aie-translate -aie-cert-to-asm %s | FileCheck %s

module {
  aie.device(npu3) {
    aiex.cert.job(7) {
      // Seed two known words through the plain immediate path.
      // CHECK: WRITE_32               0x00600400, 0x11112222
      aiex.cert.write32(0x600400, 0x11112222)
      // CHECK-NEXT: WRITE_32               0x00600404, 0x33334444
      aiex.cert.write32(0x600404, 0x33334444)

      // Pointer table for the register-held addresses below.
      // CHECK-NEXT: WRITE_32               0x00600420, 0x00600404
      aiex.cert.write32(0x600420, 0x600404)
      // CHECK-NEXT: WRITE_32               0x00600424, 0x00600410
      aiex.cert.write32(0x600424, 0x600410)
      // CHECK-NEXT: WRITE_32               0x00600428, 0x00600414
      aiex.cert.write32(0x600428, 0x600414)

      // READ_32 into $r0, echoed out with an immediate address and a
      // register-held value -> flags 2.
      // CHECK-NEXT: READ_32                $r0, 0x00600400
      aiex.cert.read32(0, 0x600400)
      // CHECK-NEXT: WRITE_32_D             2, 0x00600408, 0x00000000
      aiex.cert.write32_d(0x600408, 0) {value_is_reg}

      // READ_32_D through an address register into $r2, same echo form.
      // CHECK-NEXT: READ_32                $r1, 0x00600420
      aiex.cert.read32(1, 0x600420)
      // CHECK-NEXT: READ_32_D              $r1, $r2
      aiex.cert.read32_d(1, 2)
      // CHECK-NEXT: WRITE_32_D             2, 0x0060040c, 0x00000002
      aiex.cert.write32_d(0x60040C, 2) {value_is_reg}

      // Both operands in registers -> flags 3.
      // CHECK-NEXT: READ_32                $r3, 0x00600424
      aiex.cert.read32(3, 0x600424)
      // CHECK-NEXT: WRITE_32_D             3, 0x00000003, 0x00000001
      aiex.cert.write32_d(3, 1) {address_is_reg, value_is_reg}

      // Register-held address, immediate value -> flags 1.
      // CHECK-NEXT: READ_32                $r4, 0x00600428
      aiex.cert.read32(4, 0x600428)
      // CHECK-NEXT: WRITE_32_D             1, 0x00000004, 0x55556666
      aiex.cert.write32_d(4, 0x55556666) {address_is_reg}

      // Both operands immediate -> flags 0. This is the encoding the old
      // inverted emitter turned into "both are registers".
      // CHECK-NEXT: WRITE_32_D             0, 0x00600418, 0x77778888
      aiex.cert.write32_d(0x600418, 0x77778888)

      // CHECK-NEXT: WRITE_32               0x006ae000, 0x00000001
      aiex.cert.write32(0x6AE000, 1)
    }
  }
}

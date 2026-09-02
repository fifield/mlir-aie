//===- cert_write32_d_flags.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The ISA flags byte encodes is-REGISTER, matching the MLIR attributes
// directly (confirmed against the CERT firmware, which contradicts
// isa-spec.yaml's prose but matches the spec's own worked example):
// flags = (address_is_reg?1:0) | (value_is_reg?2:0). Getting this backwards
// produces an instruction the assembler accepts but that hangs the uC (a
// wrong-polarity flags byte makes firmware read an out-of-range "register"
// and issue a store to an unmapped address), so all four combinations are
// pinned here.

// RUN: aie-translate -aie-cert-to-asm %s | FileCheck %s

module {
  aie.device(npu2) {
    aiex.cert.job(1) {
      //   both immediate            -> flags 0
      // CHECK: WRITE_32_D             0, 0x04100000, 0x00000005
      aiex.cert.write32_d(68157440, 5)
      //   address_is_reg            -> flags 1
      // CHECK: WRITE_32_D             1, 0x00000003, 0x00000005
      aiex.cert.write32_d(3, 5) {address_is_reg}
      //   value_is_reg              -> flags 2
      // CHECK: WRITE_32_D             2, 0x04100000, 0x00000004
      aiex.cert.write32_d(68157440, 4) {value_is_reg}
      //   both registers            -> flags 3
      // CHECK: WRITE_32_D             3, 0x00000003, 0x00000004
      aiex.cert.write32_d(3, 4) {address_is_reg, value_is_reg}
    }
  }
}

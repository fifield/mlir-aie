//===- cert_verify_write32d_reg.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// G-write32d: when cert.write32_d's address_is_reg/value_is_reg is set, the
// corresponding const:32 field must be a register index in [0, 23]. This is a
// conditional check (the field's meaning depends on a sibling attribute), so
// it lives in aie-cert-verify rather than an ODS range on the attribute
// itself -- an ODS bound would wrongly reject the legitimate immediate form.

// RUN: aie-opt %s --aie-cert-verify --split-input-file --verify-diagnostics

// Positive: both immediate. 0x4100000 (68157440) is far above 23 and must NOT
// be flagged -- this is the case that proves the check is conditional, not a
// blanket ODS bound.
aie.device(npu2) {
  aiex.cert.job(0) {
    aiex.cert.write32_d(68157440, 5)
  }
}

// -----

// Positive: both register, at the in-range extreme 23.
aie.device(npu2) {
  aiex.cert.job(0) {
    aiex.cert.write32_d(23, 23) {address_is_reg, value_is_reg}
  }
}

// -----

// Negative: address_is_reg with address 24, one past the maximum.
aie.device(npu2) {
  aiex.cert.job(0) {
    // expected-error@+1 {{cert.write32_d has address_is_reg, so address 24 must be a register index in [0, 23]}}
    aiex.cert.write32_d(24, 5) {address_is_reg}
  }
}

// -----

// Negative: value_is_reg with value 24, one past the maximum.
aie.device(npu2) {
  aiex.cert.job(0) {
    // expected-error@+1 {{cert.write32_d has value_is_reg, so value 24 must be a register index in [0, 23]}}
    aiex.cert.write32_d(0, 24) {value_is_reg}
  }
}

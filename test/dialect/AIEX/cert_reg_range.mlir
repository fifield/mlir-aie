//===- cert_reg_range.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// AIEX_CertRegisterAttr bounds a source register index to [0, 23]
// (isa-spec.yaml); AIEX_CertLocalRegisterAttr bounds a destination register
// to [0, 7]. These are ODS ConfinedAttrs, so they fire without any pass -- a
// plain aie-opt parse is enough.

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Positive: the in-range extremes, 0 and 7 for a destination and 23 for a
// source.
aie.device(npu2) {
  aiex.cert.job(0) {
    aiex.cert.read32(0, 1)
    aiex.cert.read32(7, 1)
    aiex.cert.read32_d(23, 0)
  }
}

// -----

// Negative: 8 is one past the destination maximum.
aie.device(npu2) {
  aiex.cert.job(0) {
    // expected-error@+1 {{attribute 'value' failed to satisfy constraint}}
    aiex.cert.read32(8, 1)
  }
}

// -----

// Negative: 24 is one past the source maximum, for cert.read32_d's
// `address` operand.
aie.device(npu2) {
  aiex.cert.job(0) {
    // expected-error@+1 {{attribute 'address' failed to satisfy constraint}}
    aiex.cert.read32_d(24, 0)
  }
}

// -----

// Negative: cert.wait_uc_dma's `wait_handle` is bounded like a destination.
aie.device(npu2) {
  aiex.cert.job(0) {
    // expected-error@+1 {{attribute 'wait_handle' failed to satisfy constraint}}
    aiex.cert.wait_uc_dma(8)
  }
}

// -----

// Negative: -1 is below the minimum (IntMinValue<0>).
aie.device(npu2) {
  aiex.cert.job(0) {
    // expected-error@+1 {{attribute 'value' failed to satisfy constraint}}
    aiex.cert.read32(-1, 1)
  }
}

//===- cert_reg_range.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// AIEX_CertRegisterAttr bounds a job-runner register index to [0, 23]
// (isa-spec.yaml). This is an ODS ConfinedAttr, so it fires without any
// pass -- a plain aie-opt parse is enough.

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Positive: the in-range extremes, 0 and 23.
aie.device(npu2) {
  aiex.cert.job(0) {
    aiex.cert.mov(0, 1)
    aiex.cert.mov(23, 1)
  }
}

// -----

// Negative: 24 is one past the maximum.
aie.device(npu2) {
  aiex.cert.job(0) {
    // expected-error@+1 {{attribute 'dest' failed to satisfy constraint}}
    aiex.cert.mov(24, 1)
  }
}

// -----

// Negative: same bound applies to cert.read32_d's `address` operand.
aie.device(npu2) {
  aiex.cert.job(0) {
    // expected-error@+1 {{attribute 'address' failed to satisfy constraint}}
    aiex.cert.read32_d(24, 0)
  }
}

// -----

// Negative: same bound applies to cert.wait_uc_dma's `wait_handle` operand.
aie.device(npu2) {
  aiex.cert.job(0) {
    // expected-error@+1 {{attribute 'wait_handle' failed to satisfy constraint}}
    aiex.cert.wait_uc_dma(24)
  }
}

// -----

// Negative: -1 is below the minimum (IntMinValue<0>).
aie.device(npu2) {
  aiex.cert.job(0) {
    // expected-error@+1 {{attribute 'dest' failed to satisfy constraint}}
    aiex.cert.add(-1, 1)
  }
}

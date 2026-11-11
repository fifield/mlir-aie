//===- cert_verify_apply_offset_57.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// G-applyoffset: cert.apply_offset_57 rejects a num_entries of 0, and its
// unsigned fields reject negative spellings at parse time.

// RUN: aie-opt %s --aie-cert-verify --split-input-file --verify-diagnostics

// Negative: num_entries of 0 patches nothing.
module {
  aie.device(npu3) {
    memref.global "private" constant @bd0 : memref<9xi32> = dense<0>
    aiex.cert.job(0) {
      // expected-error @+1 {{cert.apply_offset_57 num_entries must be at least 1}}
      aiex.cert.apply_offset_57(@bd0, 0, 2)
    }
  }
}

// -----

// Negative: num_entries is unsigned, so a negative spelling is rejected at
// parse time. The >= 1 check above reads the zero-extended getter and would
// see -1 as 65535, i.e. a runaway patch over 65535 BD sets, so the attribute
// type -- not the verifier -- is what has to reject this.
module {
  aie.device(npu3) {
    memref.global "private" constant @bd0 : memref<9xi32> = dense<0>
    aiex.cert.job(0) {
      // expected-error @+1 {{negative integer literal not valid for unsigned integer type}}
      aiex.cert.apply_offset_57(@bd0, -1, 2)
    }
  }
}

// -----

// Positive: the PATCH_CONTROL_CODE_FLAG sentinel is accepted.
module {
  aie.device(npu3) {
    memref.global "private" constant @bd0 : memref<9xi32> = dense<0>
    aiex.cert.job(0) {
      aiex.cert.apply_offset_57(@bd0, 1, 0xffff)
    }
  }
}

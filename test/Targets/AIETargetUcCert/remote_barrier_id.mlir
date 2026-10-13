//===- remote_barrier_id.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// remote_barrier rbN is emitted as the assembler operand $rbN.

// RUN: aie-translate %s -aie-cert-to-asm | FileCheck %s
// CHECK: REMOTE_BARRIER $rb0, 3
// CHECK: REMOTE_BARRIER $rb6, 3

module {
  aie.device(npu3) {
    aiex.cert.job(1) {
      aiex.cert.remote_barrier(rb0, 0x3)
      aiex.cert.remote_barrier(rb6, 0x3)
    }
  }
}

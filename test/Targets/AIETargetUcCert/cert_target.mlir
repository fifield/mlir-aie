//===- cert_target.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The .target directive names the architecture the control code runs on; its
// optional sub-arch distinguishes architecture variants. It is omitted
// entirely for architectures with no CERT target string.

// RUN: aie-translate -aie-cert-to-asm --split-input-file %s | FileCheck %s

module {
  aie.device(npu3) {
    aiex.cert.job(0) {
      aiex.cert.nop
    }
  }
}

// -----

module {
  aie.device(xcve3858) {
    aiex.cert.job(0) {
      aiex.cert.nop
    }
  }
}

// -----

module {
  aie.device(npu2) {
    aiex.cert.job(0) {
      aiex.cert.nop
    }
  }
}

// CHECK: .target aie4
// CHECK-NEXT: .partition 3column
// CHECK: .target aie2ps
// CHECK-NEXT: .partition 36column
// CHECK-NOT: .target

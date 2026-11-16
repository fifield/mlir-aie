// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt -aie-npu-to-cert -cert-legalize-pages -aie-cert-verify --split-input-file %s | FileCheck %s

// Lowering chooses max+1 (65) for the sequence. Its slot collides with job 1
// on the implicit page, even though job 64 is on a separate page.
// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.job(64)
// CHECK: aiex.cert.page {
// CHECK-NEXT: aiex.cert.job(1)
// CHECK: aiex.cert.job(0)
// CHECK-NEXT: aiex.cert.write32(4096, 7)
aie.device(npu3) @main {
  aiex.cert.page {
    aiex.cert.job(64) {}
  }
  aiex.cert.job(1) {}
  aie.runtime_sequence @work() {
    %a = arith.constant 4096 : i32
    %v = arith.constant 7 : i32
    aiex.npu.write32(%a, %v) : i32, i32
  }
}

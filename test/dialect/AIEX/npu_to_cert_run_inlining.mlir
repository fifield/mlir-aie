//===- npu_to_cert_run_inlining.mlir ---------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-npu-to-cert %s | FileCheck %s

// Test that aiex.run operations inside configure blocks are inlined

module {
  aie.device(npu2) @main {
    // CHECK-LABEL: aiex.cert.job(1)
    // CHECK-NOT: aiex.run
    // CHECK: aiex.cert.write32(4096, 1)
    // CHECK: aiex.cert.write32(4096, 42)
    // CHECK-NOT: aiex.run
    aie.runtime_sequence @configure(%arg0: memref<16xi32>) {
      %wa0 = arith.constant 4096 : i32
      %wv0 = arith.constant 1 : i32
      aiex.npu.write32(%wa0, %wv0) {column = 0 : i32, row = 0 : i32} : i32, i32
      aiex.configure @config_device {
        aiex.run @setup(%arg0) : (memref<16xi32>)
      }
    }
  }

  aie.device(npu2) @config_device {
    aie.runtime_sequence @setup(%buf: memref<16xi32>) {
      %wa1 = arith.constant 4096 : i32
      %wv1 = arith.constant 42 : i32
      aiex.npu.write32(%wa1, %wv1) {column = 0 : i32, row = 0 : i32} : i32, i32
    }
  }
}

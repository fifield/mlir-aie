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
      aiex.npu.write32 {address = 0x1000 : ui32, column = 0 : i32, row = 0 : i32, value = 1 : ui32}
      aiex.configure @config_device {
        aiex.run @setup(%arg0) : (memref<16xi32>)
      }
    }
  }

  aie.device(npu2) @config_device {
    aie.runtime_sequence @setup(%buf: memref<16xi32>) {
      aiex.npu.write32 {address = 0x1000 : ui32, column = 0 : i32, row = 0 : i32, value = 42 : ui32}
    }
  }
}

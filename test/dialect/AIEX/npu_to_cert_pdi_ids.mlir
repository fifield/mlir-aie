//===- npu_to_cert_pdi_ids.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-npu-to-cert %s | FileCheck %s

// Test that multiple configure operations get unique PDI IDs

module {
  aie.device(npu2) @main {
    // CHECK-DAG: aiex.cert.load_pdi({{[12]}}, @config_device_a)
    // CHECK-DAG: aiex.cert.load_pdi({{[12]}}, @config_device_b)
    // CHECK-DAG: aiex.cert.section @config_device_a
    // CHECK-DAG: aiex.cert.section @config_device_b
    aie.runtime_sequence @main_seq(%arg0: memref<1024xi32>) {
      aiex.configure @config_device_a {
        aiex.run @setup_a(%arg0) : (memref<1024xi32>)
      }
      aiex.configure @config_device_b {
        aiex.run @setup_b(%arg0) : (memref<1024xi32>)
      }
    }
  }

  aie.device(npu2) @config_device_a {
    aie.runtime_sequence @setup_a(%buf: memref<1024xi32>) {
      %wa0 = arith.constant 6750208 : i32
      %wv0 = arith.constant 10 : i32
      aiex.npu.write32(%wa0, %wv0) {column = 0 : i32, row = 0 : i32} : i32, i32
    }
  }

  aie.device(npu2) @config_device_b {
    aie.runtime_sequence @setup_b(%buf: memref<1024xi32>) {
      %wa1 = arith.constant 6750208 : i32
      %wv1 = arith.constant 20 : i32
      aiex.npu.write32(%wa1, %wv1) {column = 0 : i32, row = 0 : i32} : i32, i32
    }
  }
}

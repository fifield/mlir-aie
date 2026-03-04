//===- npu_to_cert_pdi_ids_three.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-npu-to-cert %s | FileCheck %s

// Test that three configure operations get unique PDI IDs and job IDs

module {
  

  aie.device(npu2) @config_device_a {
    %tile_0_2 = aie.tile(0, 2)
    %lock_0_2 = aie.lock(%tile_0_2, 0) {init = 1 : i32}
    aie.runtime_sequence @setup_a(%buf: memref<1024xi32>) {
      aiex.npu.write32 {address = 0x8000 : ui32, column = 0 : i32, row = 2 : i32, value = 10 : ui32}
    }
  }

  aie.device(npu2) @config_device_b {
    %tile_0_3 = aie.tile(0, 3)
    %lock_0_3 = aie.lock(%tile_0_3, 1) {init = 2 : i32}
    aie.runtime_sequence @setup_b(%buf: memref<1024xi32>) {
      aiex.npu.write32 {address = 0x9000 : ui32, column = 0 : i32, row = 3 : i32, value = 20 : ui32}
    }
  }

  aie.device(npu2) @config_device_c {
    %tile_0_4 = aie.tile(0, 4)
    %lock_0_4 = aie.lock(%tile_0_4, 2) {init = 3 : i32}
    aie.runtime_sequence @setup_c(%buf: memref<1024xi32>) {
      aiex.npu.write32 {address = 0xA000 : ui32, column = 0 : i32, row = 4 : i32, value = 30 : ui32}
    }
  }
  aie.device (npu2) @main {
    // CHECK-DAG: aiex.cert.load_pdi({{[123]}}, @config_device_a)
    // CHECK-DAG: aiex.cert.load_pdi({{[123]}}, @config_device_b)
    // CHECK-DAG: aiex.cert.load_pdi({{[123]}}, @config_device_c)
    // CHECK-DAG: aiex.cert.section @config_device_a
    // CHECK-DAG: aiex.cert.section @config_device_b
    // CHECK-DAG: aiex.cert.section @config_device_c
    aie.runtime_sequence @sequence (%arg0: memref<1024xi32>) {
      aiex.configure @config_device_a {
        aiex.run @setup_a(%arg0) : (memref<1024xi32>)
      }
      aiex.configure @config_device_b {
        aiex.run @setup_b(%arg0) : (memref<1024xi32>)
      }
      aiex.configure @config_device_c {
        aiex.run @setup_c(%arg0) : (memref<1024xi32>)
      }
    }
  }
}

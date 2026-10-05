//===- lower_set_lock_aie4.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-set-lock %s | FileCheck %s

// Verify AIE4 getLocalLockAddress produces correct addresses for all tile
// types and both lock banks.

module @test_aie4_lock_addresses {
  aie.device(npu3) {
    // -- Compute tile (row 2): 16 locks, base 0xAE000, stride 0x10 --
    %tile02 = aie.tile(0, 2)
    %lock_ct_0  = aie.lock(%tile02, 0)
    %lock_ct_15 = aie.lock(%tile02, 15)

    // -- Memtile (row 1): 64 locks, bank A 0x10000 (0-31), bank B 0x50000 (32-63) --
    %tile01 = aie.tile(0, 1)
    %lock_mt_0  = aie.lock(%tile01, 0)
    %lock_mt_31 = aie.lock(%tile01, 31)
    %lock_mt_32 = aie.lock(%tile01, 32)
    %lock_mt_63 = aie.lock(%tile01, 63)

    // -- Shim tile (row 0): 32 locks, bank A 0x10000 (0-15), bank B 0x50000 (16-31) --
    %tile00 = aie.tile(0, 0)
    %lock_sh_0  = aie.lock(%tile00, 0)
    %lock_sh_15 = aie.lock(%tile00, 15)
    %lock_sh_16 = aie.lock(%tile00, 16)
    %lock_sh_31 = aie.lock(%tile00, 31)

    %core02 = aie.core(%tile02) {
      aie.end
    }

    aie.shim_dma_allocation @in0 (%tile00, S2MM, 0)

    aie.runtime_sequence(%arg0: memref<1024xi32>) {
      // Compute tile lock 0:  0xAE000 = 712704
      // CHECK-DAG: %[[V0:.*]] = arith.constant 1 : i32
      // CHECK-DAG: %[[A0:.*]] = arith.constant 712704 : i32
      // CHECK: aiex.npu.write32(%[[A0]], %[[V0]]) {column = 0 : i32, row = 2 : i32} : i32, i32
      %lock_ct_0_v1 = arith.constant 1 : i32
      aiex.set_lock(%lock_ct_0, %lock_ct_0_v1)
      // Compute tile lock 15: 0xAE0F0 = 712944
      // CHECK-DAG: %[[V1:.*]] = arith.constant 0 : i32
      // CHECK-DAG: %[[A1:.*]] = arith.constant 712944 : i32
      // CHECK: aiex.npu.write32(%[[A1]], %[[V1]]) {column = 0 : i32, row = 2 : i32} : i32, i32
      %lock_ct_15_v2 = arith.constant 0 : i32
      aiex.set_lock(%lock_ct_15, %lock_ct_15_v2)

      // Memtile lock 0:  0x10000 = 65536
      // CHECK-DAG: %[[V2:.*]] = arith.constant 1 : i32
      // CHECK-DAG: %[[A2:.*]] = arith.constant 65536 : i32
      // CHECK: aiex.npu.write32(%[[A2]], %[[V2]]) {column = 0 : i32, row = 1 : i32} : i32, i32
      %lock_mt_0_v3 = arith.constant 1 : i32
      aiex.set_lock(%lock_mt_0, %lock_mt_0_v3)
      // Memtile lock 31: 0x101F0 = 66032
      // CHECK-DAG: %[[V3:.*]] = arith.constant 0 : i32
      // CHECK-DAG: %[[A3:.*]] = arith.constant 66032 : i32
      // CHECK: aiex.npu.write32(%[[A3]], %[[V3]]) {column = 0 : i32, row = 1 : i32} : i32, i32
      %lock_mt_31_v4 = arith.constant 0 : i32
      aiex.set_lock(%lock_mt_31, %lock_mt_31_v4)
      // Memtile lock 32: 0x50000 = 327680
      // CHECK-DAG: %[[V4:.*]] = arith.constant 1 : i32
      // CHECK-DAG: %[[A4:.*]] = arith.constant 327680 : i32
      // CHECK: aiex.npu.write32(%[[A4]], %[[V4]]) {column = 0 : i32, row = 1 : i32} : i32, i32
      %lock_mt_32_v5 = arith.constant 1 : i32
      aiex.set_lock(%lock_mt_32, %lock_mt_32_v5)
      // Memtile lock 63: 0x501F0 = 328176
      // CHECK-DAG: %[[V5:.*]] = arith.constant 0 : i32
      // CHECK-DAG: %[[A5:.*]] = arith.constant 328176 : i32
      // CHECK: aiex.npu.write32(%[[A5]], %[[V5]]) {column = 0 : i32, row = 1 : i32} : i32, i32
      %lock_mt_63_v6 = arith.constant 0 : i32
      aiex.set_lock(%lock_mt_63, %lock_mt_63_v6)

      // Shim lock 0:  0x10000 = 65536
      // CHECK-DAG: %[[V6:.*]] = arith.constant 1 : i32
      // CHECK-DAG: %[[A6:.*]] = arith.constant 65536 : i32
      // CHECK: aiex.npu.write32(%[[A6]], %[[V6]]) {column = 0 : i32, row = 0 : i32} : i32, i32
      %lock_sh_0_v7 = arith.constant 1 : i32
      aiex.set_lock(%lock_sh_0, %lock_sh_0_v7)
      // Shim lock 15: 0x100F0 = 65776
      // CHECK-DAG: %[[V7:.*]] = arith.constant 0 : i32
      // CHECK-DAG: %[[A7:.*]] = arith.constant 65776 : i32
      // CHECK: aiex.npu.write32(%[[A7]], %[[V7]]) {column = 0 : i32, row = 0 : i32} : i32, i32
      %lock_sh_15_v8 = arith.constant 0 : i32
      aiex.set_lock(%lock_sh_15, %lock_sh_15_v8)
      // Shim lock 16: 0x50000 = 327680
      // CHECK-DAG: %[[V8:.*]] = arith.constant 1 : i32
      // CHECK-DAG: %[[A8:.*]] = arith.constant 327680 : i32
      // CHECK: aiex.npu.write32(%[[A8]], %[[V8]]) {column = 0 : i32, row = 0 : i32} : i32, i32
      %lock_sh_16_v9 = arith.constant 1 : i32
      aiex.set_lock(%lock_sh_16, %lock_sh_16_v9)
      // Shim lock 31: 0x500F0 = 327920
      // CHECK-DAG: %[[V9:.*]] = arith.constant 0 : i32
      // CHECK-DAG: %[[A9:.*]] = arith.constant 327920 : i32
      // CHECK: aiex.npu.write32(%[[A9]], %[[V9]]) {column = 0 : i32, row = 0 : i32} : i32, i32
      %lock_sh_31_v10 = arith.constant 0 : i32
      aiex.set_lock(%lock_sh_31, %lock_sh_31_v10)
    }
  }
}

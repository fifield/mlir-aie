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
      // CHECK: aiex.npu.write32 {address = 712704 : ui32, column = 0 : i32, row = 2 : i32, value = 1 : ui32}
      aiex.set_lock(%lock_ct_0, 1)
      // Compute tile lock 15: 0xAE0F0 = 712944
      // CHECK: aiex.npu.write32 {address = 712944 : ui32, column = 0 : i32, row = 2 : i32, value = 0 : ui32}
      aiex.set_lock(%lock_ct_15, 0)

      // Memtile lock 0:  0x10000 = 65536
      // CHECK: aiex.npu.write32 {address = 65536 : ui32, column = 0 : i32, row = 1 : i32, value = 1 : ui32}
      aiex.set_lock(%lock_mt_0, 1)
      // Memtile lock 31: 0x101F0 = 66032
      // CHECK: aiex.npu.write32 {address = 66032 : ui32, column = 0 : i32, row = 1 : i32, value = 0 : ui32}
      aiex.set_lock(%lock_mt_31, 0)
      // Memtile lock 32: 0x50000 = 327680
      // CHECK: aiex.npu.write32 {address = 327680 : ui32, column = 0 : i32, row = 1 : i32, value = 1 : ui32}
      aiex.set_lock(%lock_mt_32, 1)
      // Memtile lock 63: 0x501F0 = 328176
      // CHECK: aiex.npu.write32 {address = 328176 : ui32, column = 0 : i32, row = 1 : i32, value = 0 : ui32}
      aiex.set_lock(%lock_mt_63, 0)

      // Shim lock 0:  0x10000 = 65536
      // CHECK: aiex.npu.write32 {address = 65536 : ui32, column = 0 : i32, row = 0 : i32, value = 1 : ui32}
      aiex.set_lock(%lock_sh_0, 1)
      // Shim lock 15: 0x100F0 = 65776
      // CHECK: aiex.npu.write32 {address = 65776 : ui32, column = 0 : i32, row = 0 : i32, value = 0 : ui32}
      aiex.set_lock(%lock_sh_15, 0)
      // Shim lock 16: 0x50000 = 327680
      // CHECK: aiex.npu.write32 {address = 327680 : ui32, column = 0 : i32, row = 0 : i32, value = 1 : ui32}
      aiex.set_lock(%lock_sh_16, 1)
      // Shim lock 31: 0x500F0 = 327920
      // CHECK: aiex.npu.write32 {address = 327920 : ui32, column = 0 : i32, row = 0 : i32, value = 0 : ui32}
      aiex.set_lock(%lock_sh_31, 0)
    }
  }
}

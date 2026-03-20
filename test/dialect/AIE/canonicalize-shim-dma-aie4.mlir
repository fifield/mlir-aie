//===- canonicalize-shim-dma-aie4.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --canonicalize %s | FileCheck %s

// Regression test for the broadcast_twos BD lowering bug (74a661edae).
//
// On AIE4/AIE4z, each shim DMA channel has a private BD pool (16 slots).
// FoldDMAStartOp must NOT fold repeating BD chains in shim DMAs because
// after block merging, two channels can end up sharing the same BD blocks.
// If those blocks are then folded, the lowering generates fewer hardware
// BD writes than needed, causing channels to race for the same BD slots
// and deadlock.
//
// This test has a single channel with a repeating [bd0, bd1, bd0, bd1]
// pattern. Without the fix, FoldDMAStartOp folds it to [bd0, bd1] (2 BDs).
// With the fix, all 4 BDs are preserved.

// CHECK-LABEL: module @test_aie4_shim_dma_no_fold
// CHECK:       aie.shim_dma
// CHECK-COUNT-4: aie.dma_bd
// CHECK:       aie.end
module @test_aie4_shim_dma_no_fold {
  aie.device(npu3z) {
    %buf0 = aie.external_buffer {address = 2147483648 : i64} : memref<64xi32>
    %buf1 = aie.external_buffer {address = 2147484160 : i64} : memref<64xi32>
    %shim = aie.tile(0, 0)
    %lock0 = aie.lock(%shim, 0) {init = 0 : i32}
    %lock1 = aie.lock(%shim, 1) {init = 0 : i32}

    %shim_dma = aie.shim_dma(%shim) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lock0, Acquire, 1)
      aie.dma_bd(%buf0 : memref<64xi32>, 0, 64, [<size = 64, stride = 1>]) {bd_id = 0 : i32}
      aie.use_lock(%lock1, Release, 1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%lock0, Acquire, 1)
      aie.dma_bd(%buf1 : memref<64xi32>, 0, 64, [<size = 64, stride = 1>]) {bd_id = 1 : i32}
      aie.use_lock(%lock1, Release, 1)
      aie.next_bd ^bd2
    ^bd2:
      aie.use_lock(%lock0, Acquire, 1)
      aie.dma_bd(%buf0 : memref<64xi32>, 0, 64, [<size = 64, stride = 1>]) {bd_id = 0 : i32}
      aie.use_lock(%lock1, Release, 1)
      aie.next_bd ^bd3
    ^bd3:
      aie.use_lock(%lock0, Acquire, 1)
      aie.dma_bd(%buf1 : memref<64xi32>, 0, 64, [<size = 64, stride = 1>]) {bd_id = 1 : i32}
      aie.use_lock(%lock1, Release, 1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}

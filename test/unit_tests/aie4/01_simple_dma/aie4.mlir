//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices Inc.
//
// RUN: aie-opt -convert-aie-to-transaction %s -o txn.mlir 
// RUN: FileCheck --input-file=txn.mlir %s
//
// expect:
// blockwrite tile DMA_BD0_0 0x6AC000
// blockwrite tile DMA_BD1_0 0x6AC020
// write32 tile DMA_S2MM_0_Start_Queue 0x6ACE04
// block write shim DMA_BD0_0 0x18000
// block write shim DMA_BD1_0 0x18030
// write32 shim DMA_MM2S_0_Task_Queue 0x18494
//
// TODO: verify the data being written then add it to the checks
//
// CHECK-LABEL: aie.runtime_sequence @configure()
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 6995968 : ui32} : memref<6xi32>
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 6996000 : ui32} : memref<6xi32>
// CHECK: aiex.npu.write32 {address = 6999556 : ui32, value = 0 : ui32}
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 98304 : ui32} : memref<9xi32>
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 98352 : ui32} : memref<9xi32>
// CHECK: aiex.npu.write32 {address = 99668 : ui32, value = 0 : ui32}
//
module @test_dma_to_txn {
  aie.device(npu3) {
    %tile_0_2 = aie.tile(0, 2)
    %buf02_0 = aie.buffer(%tile_0_2) {address = 2048 : i32, mem_bank = 0 : i32, sym_name = "buf02_0"} : memref<256xi32>
    %buf02_1 = aie.buffer(%tile_0_2) {address = 17408 : i32, mem_bank = 1 : i32, sym_name = "buf02_1"} : memref<256xi32>
    %lock_0_2 = aie.lock(%tile_0_2, 0) { init = 2 : i32 }
    %lock_0_2_0 = aie.lock(%tile_0_2, 1)
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    }

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf02_0 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_0_2_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf02_1 : memref<256xi32>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }

    %tile_0_0 = aie.tile(0, 0)
    %buf00_0 = aie.external_buffer {address = 0x80000000 : i64, sym_name = "buf00_0"} : memref<256 x i32>
    %buf00_1 = aie.external_buffer {address = 0x80000400 : i64, sym_name = "buf00_1"} : memref<256 x i32>
    %lock_0_0 = aie.lock(%tile_0_0, 0) { init = 2 : i32 }
    %lock_0_0_0 = aie.lock(%tile_0_0, 1)
    %mem_0_0 = aie.shim_dma(%tile_0_0) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf00_0 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_0_0_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf00_1 : memref<256xi32>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_0_0_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
  }
}

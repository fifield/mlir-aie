//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices Inc.
//
// RUN: aiecc.py --xchesscc --xbridge --no-compile-host --aie-generate-txn --aiesim %s %test_utils_flags %S/test.cpp
// RUN: aie-translate -aie-npu-instgen -aie-output-binary=true -o transaction.bin aie4.mlir.prj/txn.mlir
// RUN: aie4.mlir.prj/aiesim.sh | FileCheck %s
//
// CHECK: AIE4{{.*}}ISS
// CHECK: PASS!
//
module @test_dma_to_txn {
  aie.device(npu3) {
    %tile_0_2 = aie.tile(0, 2)
    %buf02_0 = aie.buffer(%tile_0_2) {address = 2048 : i32, mem_bank = 0 : i32, sym_name = "buf02_0"} : memref<256xi32>
    %buf02_1 = aie.buffer(%tile_0_2) {address = 4096 : i32, mem_bank = 0 : i32, sym_name = "buf02_1"} : memref<256xi32>
    %lock_0_2 = aie.lock(%tile_0_2, 0) { init = 1 : i32 }
    %lock_0_2_0 = aie.lock(%tile_0_2, 1)
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    }

    %tile_0_1 = aie.tile(0, 1)
    %mem_0_1 = aie.memtile_dma(%tile_0_1) {
      aie.end
    }

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
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
    %buf00_0 = aie.external_buffer {address = 0 : i32, sym_name = "buf00_0"} : memref<256 x i32>
    %buf00_1 = aie.external_buffer {address = 0 : i32, sym_name = "buf00_1"} : memref<256 x i32>
    %lock_0_0 = aie.lock(%tile_0_0, 0) { init = 1 : i32 }
    %lock_0_0_0 = aie.lock(%tile_0_0, 1)
    %mem_0_0 = aie.shim_dma(%tile_0_0) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf00_0 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_0_0_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf00_0 : memref<256xi32>, 0, 256) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%lock_0_0_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }

    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)
  }
}

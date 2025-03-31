//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices Inc.
//
// REQUIRES: mesimulator
//
// RUN: aiecc.py --alloc-scheme=basic-sequential --xchesscc --xbridge --no-compile-host --aie-generate-txn --aiesim %s %test_utils_flags %S/test.cpp
// RUN: aie-translate -aie-npu-to-binary -o transaction.bin aie4z.mlir.prj/txn.mlir
// RUN: aie4z.mlir.prj/mesim.sh | FileCheck --check-prefix=MESIM %s
//
// MESIM: device model: {{.*}}aie4z
// MESIM: PASS!
//
module @test_dma_to_txn {
  aie.device(npu3z) {
    %tile_0_2 = aie.tile(0, 3)
    %buf02_0 = aie.buffer(%tile_0_2) {address = 2048 : i32, bank = 0 : i32, sym_name = "buf02_0"} : memref<256xi32>
    %buf02_1 = aie.buffer(%tile_0_2) {address = 4096 : i32, bank = 0 : i32, sym_name = "buf02_1"} : memref<256xi32>
    %lock_0_2 = aie.lock(%tile_0_2, 0) { init = 2 : i32 }
    %lock_0_2_0 = aie.lock(%tile_0_2, 1) { init = 0 : i32 }

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
    // %core_0_2 = aie.core(%tile_0_2) {
    //   aie.use_lock(%lock_0_2, AcquireGreaterEqual, 3)
    //   aie.end
    // }

    %tile_0_4 = aie.tile(0, 4)
    %buf04_0 = aie.buffer(%tile_0_4) {address = 2048 : i32, bank = 0 : i32, sym_name = "buf04_0"} : memref<256xi32>
    %buf04_1 = aie.buffer(%tile_0_4) {address = 4096 : i32, bank = 0 : i32, sym_name = "buf04_1"} : memref<256xi32>

    %lock_0_4 = aie.lock(%tile_0_4, 0) { init = 2 : i32 }
    %lock_0_4_0 = aie.lock(%tile_0_4, 1)

    %mem_0_1 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf04_0 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_0_4_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf04_1 : memref<256xi32>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_0_4_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
    // %core_0_4 = aie.core(%tile_0_4) {
    //   aie.use_lock(%lock_0_4, AcquireGreaterEqual, 3)
    //   aie.end
    // }

    aie.flow(%tile_0_4, DMA : 0, %tile_0_2, DMA : 0)
  }
}

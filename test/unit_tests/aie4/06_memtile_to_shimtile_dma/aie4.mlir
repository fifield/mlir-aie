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
// RUN: aie-translate -aie-npu-to-binary -aie-output-binary=true -o transaction.bin aie4.mlir.prj/txn.mlir
// RUN: aie4.mlir.prj/aiesim.sh | FileCheck --check-prefix=AIESIM %s
// RUN: aie4.mlir.prj/mesim.sh | FileCheck --check-prefix=MESIM %s
//
// AIESIM: AIE4{{.*}}ISS
// AIESIM: PASS!
// MESIM: device model: {{.*}}aie4
// MESIM: PASS!
//
module {
  aie.device(npu3) {

    %tile_0_1 = aie.tile(0, 1)
    %buf01_0 = aie.buffer(%tile_0_1) {address = 2048 : i32, mem_bank = 0 : i32, sym_name = "buf01_0"} : memref<256xi32>
    %buf01_1 = aie.buffer(%tile_0_1) {address = 4096 : i32, mem_bank = 0 : i32, sym_name = "buf01_1"} : memref<256xi32>
    %lock_0_1_0 = aie.lock(%tile_0_1, 0) { init = 1 : i32 }
    %lock_0_1_1 = aie.lock(%tile_0_1, 1)

    %mem_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf01_0 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_0_1_1, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf01_1 : memref<256xi32>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_0_1_1, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }

    %tile_0_0 = aie.tile(0, 0)
    %buf00_0 = aie.external_buffer {address = 0 : i32, sym_name = "buf00_0"} : memref<256 x i32>
    %buf00_1 = aie.external_buffer {address = 2048 : i32, sym_name = "buf00_1"} : memref<256 x i32>
    %lock_0_0 = aie.lock(%tile_0_0, 0) { init = 1 : i32 }
    %lock_0_0_0 = aie.lock(%tile_0_0, 1)
    %mem_0_0 = aie.shim_dma(%tile_0_0) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:
      aie.use_lock(%lock_0_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf00_0 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_0_0_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }

    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
  }
}

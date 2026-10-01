// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
// REQUIRES: mesimulator
//
// RUN: mkdir -p project0
// RUN: cp %aie_obj_root/third_party/cert/main/cert_medusa.elf project0/cert_medusa.elf
// RUN: aiecc --tmpdir=project0 --alloc-scheme=basic-sequential --xchesscc --xbridge --no-compile --aie-generate-txn --txn-name=transaction.bin --aiesim %s %test_utils_flags %S/test.cpp
// RUN: aiecc --tmpdir=project1 --alloc-scheme=basic-sequential --xchesscc --xbridge --no-compile --get-full-elf --full-elf-name=aie4.elf %S/cert.mlir
// RUN: project0/mesim.sh | FileCheck %s
//
// CHECK: PASS!
//
// based on unit_tests/aie4/02_memtile_to_tile_dma/aie4.mlir
// Lock 0 in Tile 1,2 is initialized to 0. It is later set to 2 by the
// runtime_sequence, unblocking the dma transfer from memtile dma to tile dma.
module {

  aie.device(npu3) {
    %tile_1_0 = aie.tile(1, 0) { elf_file = "cert_medusa.elf" }
    %tile_1_2 = aie.tile(1, 2)
    %buf12_0 = aie.buffer(%tile_1_2) {address = 2048 : i32, bank = 0 : i32, sym_name = "buf12_0"} : memref<256xi32>
    %buf12_1 = aie.buffer(%tile_1_2) {address = 4096 : i32, bank = 0 : i32, sym_name = "buf12_1"} : memref<256xi32>

    // lock 0 will be set to 2 by aiex.cert.job(1)
    %lock_1_2 = aie.lock(%tile_1_2, 0) { init = 0 : i32 }
    %lock_1_2_0 = aie.lock(%tile_1_2, 1) { init = 0 : i32 }

    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf12_0 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_1_2_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf12_1 : memref<256xi32>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_1_2_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }

    %tile_1_1 = aie.tile(1, 1)
    %buf11_0 = aie.buffer(%tile_1_1) {address = 2048 : i32, bank = 0 : i32, sym_name = "buf11_0"} : memref<256xi32>
    %buf11_1 = aie.buffer(%tile_1_1) {address = 4096 : i32, bank = 0 : i32, sym_name = "buf11_1"} : memref<256xi32>

    %lock_1_1 = aie.lock(%tile_1_1, 0) { init = 2 : i32 }
    %lock_1_1_0 = aie.lock(%tile_1_1, 1)

    %mem_0_1 = aie.memtile_dma(%tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf11_0 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_1_1_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf11_1 : memref<256xi32>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_1_1_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }

    aie.flow(%tile_1_1, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%tile_1_0, Control32 : 1, %tile_1_0, Controller32 : 0)
  }

}
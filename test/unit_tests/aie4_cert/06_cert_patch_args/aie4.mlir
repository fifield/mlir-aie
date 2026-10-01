//
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
// RUN: aiecc --tmpdir=project0 --alloc-scheme=basic-sequential --xchesscc --xbridge --no-compile --aie-generate-txn --txn-name=transaction.bin --no-compile-host --aiesim %s %test_utils_flags %S/test.cpp
// RUN: aiecc --tmpdir=project1 --alloc-scheme=basic-sequential --xchesscc --xbridge --no-compile --get-full-elf --full-elf-name=aie4.elf %S/cert.mlir
// RUN: project0/mesim.sh | FileCheck %s
//
// CHECK: PASS!
//
module {
  aie.device(npu3) {

    %tile_0_1 = aie.tile(0, 1)
    %buf01_0 = aie.buffer(%tile_0_1) {address = 2048 : i32, mem_bank = 0 : i32, sym_name = "buf01_0"} : memref<256xi32>
    %lock_0_1_0 = aie.lock(%tile_0_1, 0) { init = 1 : i32 }
    %lock_0_1_1 = aie.lock(%tile_0_1, 1)

    %mem_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf01_0 : memref<256xi32>, 0, 256) {bd_id = 0 : i32}
      aie.use_lock(%lock_0_1_1, Release, 1)
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }

    %tile_0_0 = aie.tile(0, 0) { elf_file = "cert_medusa.elf" }
    %lock_0_0 = aie.lock(%tile_0_0, 0) { init = 1 : i32 }
    aie.flow(%tile_0_0, Control32 : 0, %tile_0_0, Controller32 : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
  }
}

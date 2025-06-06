//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
// REQUIRES: mesimulator
//
// RUN: aiecc.py --tmpdir=project --alloc-scheme=basic-sequential --xchesscc --xbridge --no-compile-host --aie-generate-elf --aiesim %s %test_utils_flags %S/test.cpp
// RUN: aie-opt --convert-aie-to-transaction='elf-dir=%aie_obj_root/third_party/cert/main' %S/base.mlir -o transaction.mlir
// RUN: aie-translate -aie-npu-to-binary -o transaction.bin transaction.mlir
// RUN: project/mesim.sh | FileCheck --check-prefix=MESIM %s
//
// MESIM: device model: {{.*}}aie4
// MESIM: PASS!
//
module {
  aie.device(npu3) {
    %tile_0_1 = aie.tile(0, 1)
    %buf01_0 = aie.buffer(%tile_0_1) {address = 2048 : i32, mem_bank = 0 : i32, sym_name = "buf01_0"} : memref<256xi32>
    %lock_0_2 = aie.lock(%tile_0_1, 0) { init = 1 : i32 }
    %lock_0_2_0 = aie.lock(%tile_0_1, 1)

    %mem_0_2 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf01_0 : memref<256xi32>, 0, 256) {bd_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }

    %tile_0_0 = aie.tile(0, 0)
    %buf00_0 = aie.external_buffer {address = 0x80000000 : i64, sym_name = "buf00_0"} : memref<256 x i32>
    %lock_0_0 = aie.lock(%tile_0_0, 0) { init = 2 : i32 }
    %lock_0_0_0 = aie.lock(%tile_0_0, 1)
    %mem_0_0 = aie.shim_dma(%tile_0_0) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      aie.use_lock(%lock_0_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf00_0 : memref<256xi32>, 0, 256) {bd_id = 0 : i32}
      aie.use_lock(%lock_0_0_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }

    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
  }
}

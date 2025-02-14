//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices Inc.
//
// RUN: aiecc.py --alloc-scheme=basic-sequential --xchesscc --xbridge --no-compile-host --aie-generate-txn --aiesim %s %test_utils_flags %S/test.cpp
// RUN: aie-translate -aie-npu-to-binary -aie-output-binary=true -o transaction.bin aie4.mlir.prj/txn.mlir
// RUN: aie4.mlir.prj/aiesim.sh | FileCheck %s
//
// CHECK: AIE4{{.*}}ISS
// CHECK: PASS!
//

module @test_12_ddr_read_bw {
  aie.device(npu3) {
    %tile_0_2 = aie.tile(0, 2)

    %ping_in_buffer  = aie.buffer(%tile_0_2) {address = 0 : i32, bank = 0 : i32, sym_name = "ping_in_buffer"} : memref<1024xi32>
    %pong_in_buffer  = aie.buffer(%tile_0_2) {address = 0 : i32, bank = 1 : i32, sym_name = "pong_in_buffer"} : memref<1024xi32>

    %mem_0_2 = aie.mem(%tile_0_2) {
        %dstDma = aie.dma_start("S2MM", 0, ^bb4, ^end)
    ^bb4: 
      aie.dma_bd(%ping_in_buffer : memref<1024xi32>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.next_bd ^bb5
    ^bb5: 
      aie.dma_bd(%pong_in_buffer : memref<1024xi32>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.next_bd ^bb4
    ^end:   
      aie.end
    }

    %tile_0_0 = aie.tile(0, 0)
    %ext_input_buffer  = aie.external_buffer{address = 0 : i32, sym_name = "ext_input_buffer"}  : memref<65536xi32>

    %shim_tile_input_write_lock  = aie.lock(%tile_0_0, 0) { init = 1 : i32 } 
    %shim_tile_input_read_lock   = aie.lock(%tile_0_0, 1) { init = 0 : i32 } 

    %mem_0_1 = aie.shim_dma(%tile_0_0) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
    ^bb1:
      aie.use_lock(%shim_tile_input_read_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_input_buffer : memref<65536xi32>, 0, 65536) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%shim_tile_input_write_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb4:
      aie.end
    }
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
  }
}

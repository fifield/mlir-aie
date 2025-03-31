//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices Inc.
//
// REQUIRES: mesimulator
//
// RUN: xchesscc_wrapper aie4 -c %S/kernel.cc
// RUN: aiecc.py --alloc-scheme=basic-sequential --xchesscc --xbridge --no-compile-host --aie-generate-txn --aiesim %s %test_utils_flags %S/test.cpp
// RUN: aie-translate -aie-npu-to-binary -o transaction.bin aie4.mlir.prj/txn.mlir
// RUN: aie4.mlir.prj/mesim.sh | FileCheck --check-prefix=MESIM %s
//
// MESIM: device model: {{.*}}aie4z
// MESIM: PASS!
//
module @test_dma_to_txn {
  aie.device(npu3z) {
    %tile_0_2 = aie.tile(0, 2)

    %ping_in_buffer  = aie.buffer(%tile_0_2) {address = 2048 : i32, bank = 0 : i32, sym_name = "ping_in_buffer"} : memref<32xi32>
    %pong_in_buffer  = aie.buffer(%tile_0_2) {address = 2304 : i32, bank = 0 : i32, sym_name = "pong_in_buffer"} : memref<32xi32>
    %ping_out_buffer = aie.buffer(%tile_0_2) {address = 2560 : i32, bank = 0 : i32, sym_name = "ping_out_buffer"} : memref<32xi32>
    %pong_out_buffer = aie.buffer(%tile_0_2) {address = 2816 : i32, bank = 0 : i32, sym_name = "pong_out_buffer"} : memref<32xi32>

    %tile_input_read_lock   = aie.lock(%tile_0_2, 0) { init = 0 : i32 } 
    %tile_input_write_lock  = aie.lock(%tile_0_2, 1) { init = 1 : i32 } 
    %tile_output_read_lock  = aie.lock(%tile_0_2, 2) { init = 0 : i32 } 
    %tile_output_write_lock = aie.lock(%tile_0_2, 3) { init = 1 : i32 } 

    func.func private @add_one(%A: memref<32xi32>, %B: memref<32xi32>) -> ()

    %core02 = aie.core(%tile_0_2) {

      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index  

      scf.for %i = %c0 to %c1024 step %c1 {
        aie.use_lock(%tile_input_read_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%tile_output_write_lock, AcquireGreaterEqual, 1)
        func.call @add_one(%ping_in_buffer, %ping_out_buffer) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%tile_input_write_lock, Release, 1)
        aie.use_lock(%tile_output_read_lock, Release, 1)

        aie.use_lock(%tile_input_read_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%tile_output_write_lock, AcquireGreaterEqual, 1)
        func.call @add_one(%pong_in_buffer, %pong_out_buffer) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%tile_input_write_lock, Release, 1)
        aie.use_lock(%tile_output_read_lock, Release, 1)
      }
      aie.end
    } { link_with="kernel.o" }
      
    %mem_0_2 = aie.mem(%tile_0_2) {
        %srcDma = aie.dma_start("MM2S", 0, ^bb1, ^dma0)
      ^dma0:
        %dstDma = aie.dma_start("S2MM", 0, ^bb4, ^end)
    ^bb1: 
      aie.use_lock(%tile_output_read_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_out_buffer : memref<32xi32>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%tile_output_write_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2: 
      aie.use_lock(%tile_output_read_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_out_buffer : memref<32xi32>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%tile_output_write_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb4: 
      aie.use_lock(%tile_input_write_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_in_buffer : memref<32xi32>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%tile_input_read_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5: 
      aie.use_lock(%tile_input_write_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_in_buffer : memref<32xi32>, 0, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%tile_input_read_lock, Release, 1)
      aie.next_bd ^bb4
    ^end:   
      aie.end
    }

    %tile_0_1 = aie.tile(0, 1)
    %mem_tile_input_buffer = aie.buffer(%tile_0_1) {address = 2048 : i32, bank = 0 : i32, sym_name = "mem_tile_input_buffer"} : memref<256xi32>
    %mem_tile_output_buffer = aie.buffer(%tile_0_1) {address = 3072 : i32, bank = 0 : i32, sym_name = "mem_tile_output_buffer"} : memref<256xi32>

    %mem_tile_input_write_lock  = aie.lock(%tile_0_1, 0) { init = 1 : i32 } 
    %mem_tile_input_read_lock   = aie.lock(%tile_0_1, 1) { init = 0 : i32 } 
    %mem_tile_output_write_lock = aie.lock(%tile_0_1, 2) { init = 1 : i32 } 
    %mem_tile_output_read_lock  = aie.lock(%tile_0_1, 3) { init = 0 : i32 } 

    %mem_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb0)
    ^bb0:
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%mem_tile_input_read_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem_tile_input_buffer : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%mem_tile_input_write_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%mem_tile_output_write_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem_tile_output_buffer : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%mem_tile_output_read_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb3
      aie.end
    }
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 0)
  }
}

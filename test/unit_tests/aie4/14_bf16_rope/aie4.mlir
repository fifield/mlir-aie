//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices Inc.
//
// RUN: xchesscc_wrapper aie4 -c %S/kernel.cc
// RUN: aiecc.py --alloc-scheme=basic-sequential --xchesscc --xbridge --no-compile-host --aie-generate-txn --aiesim %s %test_utils_flags %S/test.cpp
// RUN: aie-translate -aie-npu-to-binary -aie-output-binary=true -o transaction.bin aie4.mlir.prj/txn.mlir
// RUN: aie4.mlir.prj/aiesim.sh | FileCheck --check-prefix=AIESIM %s
//
// AIESIM: AIE4{{.*}}ISS
// AIESIM: PASS!
//

module @test_14_bf16_rope {
  aie.device(npu3) {
    %tile_0_2 = aie.tile(0, 2)

    %ping_in_buffer  = aie.buffer(%tile_0_2) {address = 2048 : i32, bank = 0 : i32, sym_name = "ping_in_buffer"} : memref<64xbf16>
    %pong_in_buffer  = aie.buffer(%tile_0_2) {address = 2304 : i32, bank = 0 : i32, sym_name = "pong_in_buffer"} : memref<64xbf16>
    %ping_sincos_table  = aie.buffer(%tile_0_2) {address = 2560 : i32, bank = 0 : i32, sym_name = "ping_sincos_table"} : memref<64xbf16>
    %pong_sincos_table  = aie.buffer(%tile_0_2) {address = 2816 : i32, bank = 0 : i32, sym_name = "pong_sincos_table"} : memref<64xbf16>
    %ping_out_buffer = aie.buffer(%tile_0_2) {address = 3072 : i32, bank = 0 : i32, sym_name = "ping_out_buffer"} : memref<64xbf16>
    %pong_out_buffer = aie.buffer(%tile_0_2) {address = 3328 : i32, bank = 0 : i32, sym_name = "pong_out_buffer"} : memref<64xbf16>

    %tile_input_read_lock   = aie.lock(%tile_0_2, 0) { init = 0 : i32 } 
    %tile_input_write_lock  = aie.lock(%tile_0_2, 1) { init = 1 : i32 } 
    %tile_sincos_read_lock   = aie.lock(%tile_0_2, 2) { init = 0 : i32 } 
    %tile_sincos_write_lock  = aie.lock(%tile_0_2, 3) { init = 1 : i32 } 
    %tile_output_read_lock  = aie.lock(%tile_0_2, 4) { init = 0 : i32 } 
    %tile_output_write_lock = aie.lock(%tile_0_2, 5) { init = 1 : i32 } 

    func.func private @rope(%A: memref<64xbf16>, %B: memref<64xbf16>, %C: memref<64xbf16>) -> ()

    %core02 = aie.core(%tile_0_2) {

      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index  

      scf.for %i = %c0 to %c1024 step %c1 {
        aie.use_lock(%tile_input_read_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%tile_sincos_read_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%tile_output_write_lock, AcquireGreaterEqual, 1)
        func.call @rope(%ping_in_buffer, %ping_sincos_table, %ping_out_buffer) : (memref<64xbf16>, memref<64xbf16>, memref<64xbf16>) -> ()
        aie.use_lock(%tile_input_write_lock, Release, 1)
        aie.use_lock(%tile_sincos_write_lock, Release, 1)
        aie.use_lock(%tile_output_read_lock, Release, 1)

        aie.use_lock(%tile_input_read_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%tile_sincos_read_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%tile_output_write_lock, AcquireGreaterEqual, 1)
        func.call @rope(%pong_in_buffer, %pong_sincos_table, %pong_out_buffer) : (memref<64xbf16>, memref<64xbf16>, memref<64xbf16>) -> ()
        aie.use_lock(%tile_input_write_lock, Release, 1)
        aie.use_lock(%tile_sincos_write_lock, Release, 1)
        aie.use_lock(%tile_output_read_lock, Release, 1)
      }
      aie.end
    } { link_with="kernel.o" }
      
    %mem_0_2 = aie.mem(%tile_0_2) {
        %srcDma = aie.dma_start("MM2S", 0, ^bb2, ^dma0)
      ^dma0:
        %dstDma0 = aie.dma_start("S2MM", 0, ^bb4, ^dma1)
      ^dma1:
        %dstDma1 = aie.dma_start("S2MM", 1, ^bb6, ^end)
    ^bb2: 
      aie.use_lock(%tile_output_read_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_out_buffer : memref<64xbf16>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%tile_output_write_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb3: 
      aie.use_lock(%tile_output_read_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_out_buffer : memref<64xbf16>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%tile_output_write_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb4: 
      aie.use_lock(%tile_input_write_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_in_buffer : memref<64xbf16>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%tile_input_read_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5: 
      aie.use_lock(%tile_input_write_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_in_buffer : memref<64xbf16>, 0, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%tile_input_read_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6: 
      aie.use_lock(%tile_sincos_write_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_sincos_table : memref<64xbf16>, 0, 32) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%tile_sincos_read_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb7: 
      aie.use_lock(%tile_sincos_write_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_sincos_table : memref<64xbf16>, 0, 32) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%tile_sincos_read_lock, Release, 1)
      aie.next_bd ^bb6
    ^end:   
      aie.end
    }

    %tile_0_0 = aie.tile(0, 0)
    %input_memory  = aie.external_buffer{address = 0x80000000 : i64, sym_name = "input_memory"}  : memref<2048xi32>
    %sincos_memory  = aie.external_buffer{address = 0x80002000 : i64, sym_name = "sincos_memory"}  : memref<2048xi32>
    %output_memory  = aie.external_buffer{address = 0x80004000 : i64, sym_name = "output_memory"}  : memref<2048xi32>

    %shim_tile_input_write_lock  = aie.lock(%tile_0_0, 0) { init = 1 : i32 } 
    %shim_tile_input_read_lock   = aie.lock(%tile_0_0, 1) { init = 0 : i32 } 
    %shim_tile_sincos_write_lock  = aie.lock(%tile_0_0, 2) { init = 1 : i32 } 
    %shim_tile_sincos_read_lock   = aie.lock(%tile_0_0, 3) { init = 0 : i32 } 
    %shim_tile_output_write_lock = aie.lock(%tile_0_0, 4) { init = 1 : i32 } 
    %shim_tile_output_read_lock  = aie.lock(%tile_0_0, 5) { init = 0 : i32 } 

    %mem_0_1 = aie.shim_dma(%tile_0_0) {
      %0 = aie.dma_start(MM2S, 0, ^bb2, ^bb0)
    ^bb0:
      %1 = aie.dma_start(MM2S, 1, ^bb3, ^bb1)
    ^bb1:
      %2 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb2:  
      aie.use_lock(%shim_tile_input_read_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%input_memory : memref<2048xi32>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%shim_tile_input_write_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb3:  
      aie.use_lock(%shim_tile_sincos_read_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%sincos_memory : memref<2048xi32>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%shim_tile_sincos_write_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  
      aie.use_lock(%shim_tile_output_write_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_memory : memref<2048xi32>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%shim_tile_output_read_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb5:
      aie.end
    }
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_0, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)
  }
}

//===- aie4z.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: mesimulator
//
// RUN: xchesscc_wrapper aie4 -c %S/kernel.cc
// RUN: aiecc.py --alloc-scheme=basic-sequential --xchesscc --xbridge --no-compile-host --aie-generate-txn --aiesim %s %test_utils_flags %S/test.cpp
// RUN: aie-translate -aie-npu-to-binary -o transaction.bin aie4z.mlir.prj/txn.mlir
// RUN: aie4z.mlir.prj/mesim.sh | tee %t.log | FileCheck --check-prefix=MESIM %s
//
// MESIM: device model: {{.*}}aie4z
// MESIM: PASS
//

module @test_matmul {
  aie.device(npu3z) {
    %tile_0_2 = aie.tile(0, 2)

    %buf13_2 = aie.buffer(%tile_0_2) { sym_name = "c", address = 65536 : i32 } : memref<256xi32>

    %tile_lock_1 = aie.lock(%tile_0_2, 1) { sym_name = "input_lock", init = 0 : i32 }
    %tile_lock_3 = aie.lock(%tile_0_2, 3) { sym_name = "output_lock", init = 0 : i32  }

    func.func private @matmul(%C: memref<256xi32>) -> ()

    %core13 = aie.core(%tile_0_2) {
        %true_init = arith.constant true
        scf.while (%cond = %true_init) : (i1) -> () {
           // Condition block
           scf.condition(%cond)
        } do {
          // Body block
          ^bb0():	      
          aie.use_lock(%tile_lock_1, AcquireGreaterEqual, 1)
          func.call @matmul(%buf13_2) : (memref<256xi32>) -> ()
          aie.use_lock(%tile_lock_3, Release)
          // Loop back unconditionally
          %true_next = arith.constant true
          scf.yield %true_next : i1
       }
       aie.end
    } { link_with="kernel.o" }



    %tile_0_0 = aie.tile(0, 0)
    %A_buf = aie.external_buffer {address = 0x80000000 : i64, sym_name = "A_buf"} : memref<256 x i32>
    %B_buf = aie.external_buffer {address = 0x80000400 : i64, sym_name = "B_buf"} : memref<4096 x i32>

    %shim_a_read_lock = aie.lock(%tile_0_0, 1) { init = 0 : i32 }
    %shim_a_write_lock = aie.lock(%tile_0_0, 2)

    %shim_b_read_lock = aie.lock(%tile_0_0, 3) { init = 0 : i32 }
    %shim_b_write_lock = aie.lock(%tile_0_0, 4)
    
    %mem_0_0 = aie.shim_dma(%tile_0_0) {
      %mm2s0 = aie.dma_start(MM2S, 0, ^bb1, ^dma1)
    ^dma1:
      %mm2s1 = aie.dma_start(MM2S, 1, ^bb2, ^bb5)  
    ^bb1:
      aie.use_lock(%shim_a_read_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_buf : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%shim_a_write_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:
      aie.use_lock(%shim_b_read_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_buf : memref<4096xi32>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%shim_b_write_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb5:
      aie.end
    }

    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, Core : 0)
    aie.flow(%tile_0_0, DMA : 1, %tile_0_2, Core : 1)

  }
}

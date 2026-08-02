//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Elementwise int32 + int32 -> int32 on a single core tile.
//
// RUN: xchesscc_wrapper aie4 -c %S/kernel.cc
// RUN: aiecc --alloc-scheme=basic-sequential --xchesscc --xbridge --get-txn --txn-name=transaction.bin --get-aiesim %s %test_utils_flags -- %S/test_sim.cpp
// RUN: aie4.mlir.prj/aiesim.sh | FileCheck --check-prefix=AIESIM %s
// XFAIL: *
//
// AIESIM: AIE4{{.*}}sim
// AIESIM: PASS!
//

module @vadd_int32 {
  aie.device(npu3) {
    %c1_i32 = arith.constant 1 : i32
    %tile_0_2 = aie.tile(0, 2)

    %buf_a = aie.buffer(%tile_0_2) {address = 2048 : i32, sym_name = "buf_a"} : memref<256xi32>
    %buf_b = aie.buffer(%tile_0_2) {address = 3072 : i32, sym_name = "buf_b"} : memref<256xi32>
    %buf_c = aie.buffer(%tile_0_2) {address = 4096 : i32, sym_name = "buf_c"} : memref<256xi32>

    %lock_a_write = aie.lock(%tile_0_2, 0) { init = 1 : i32 }
    %lock_a_read  = aie.lock(%tile_0_2, 1) { init = 0 : i32 }
    %lock_b_write = aie.lock(%tile_0_2, 2) { init = 1 : i32 }
    %lock_b_read  = aie.lock(%tile_0_2, 3) { init = 0 : i32 }
    %lock_c_write = aie.lock(%tile_0_2, 4) { init = 1 : i32 }
    %lock_c_read  = aie.lock(%tile_0_2, 5) { init = 0 : i32 }

    func.func private @vadd_int32(%A: memref<256xi32>, %B: memref<256xi32>, %C: memref<256xi32>) -> ()

    %core02 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1024 = arith.constant 1024 : index
      scf.for %i = %c0 to %c1024 step %c1 {
        aie.use_lock(%lock_a_read, AcquireGreaterEqual, %c1_i32)
        aie.use_lock(%lock_b_read, AcquireGreaterEqual, %c1_i32)
        aie.use_lock(%lock_c_write, AcquireGreaterEqual, %c1_i32)
        func.call @vadd_int32(%buf_a, %buf_b, %buf_c) : (memref<256xi32>, memref<256xi32>, memref<256xi32>) -> ()
        aie.use_lock(%lock_a_write, Release, %c1_i32)
        aie.use_lock(%lock_b_write, Release, %c1_i32)
        aie.use_lock(%lock_c_read, Release, %c1_i32)
      }
      aie.end
    } { link_with="kernel.o" }

    %mem_0_2 = aie.mem(%tile_0_2) {
        %srcDma = aie.dma_start("MM2S", 0, ^out0, ^dma_a)
      ^dma_a:
        %dstDma0 = aie.dma_start("S2MM", 0, ^in_a, ^dma_b)
      ^dma_b:
        %dstDma1 = aie.dma_start("S2MM", 1, ^in_b, ^end)
      ^out0:
        aie.use_lock(%lock_c_read, AcquireGreaterEqual, %c1_i32)
        aie.dma_bd(%buf_c : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
        aie.use_lock(%lock_c_write, Release, %c1_i32)
        aie.next_bd ^out0
      ^in_a:
        aie.use_lock(%lock_a_write, AcquireGreaterEqual, %c1_i32)
        aie.dma_bd(%buf_a : memref<256xi32> offset = 0 len = 256) {bd_id = 2 : i32, next_bd_id = 2 : i32}
        aie.use_lock(%lock_a_read, Release, %c1_i32)
        aie.next_bd ^in_a
      ^in_b:
        aie.use_lock(%lock_b_write, AcquireGreaterEqual, %c1_i32)
        aie.dma_bd(%buf_b : memref<256xi32> offset = 0 len = 256) {bd_id = 4 : i32, next_bd_id = 4 : i32}
        aie.use_lock(%lock_b_read, Release, %c1_i32)
        aie.next_bd ^in_b
      ^end:
        aie.end
    }

    %tile_0_0 = aie.tile(0, 0)
    %ext_a = aie.external_buffer{address = 0x80000000 : i64, sym_name = "ext_a"} : memref<1024xi32>
    %ext_b = aie.external_buffer{address = 0x80001000 : i64, sym_name = "ext_b"} : memref<1024xi32>
    %ext_c = aie.external_buffer{address = 0x80002000 : i64, sym_name = "ext_c"} : memref<1024xi32>

    %shim_lock_a_write = aie.lock(%tile_0_0, 0) { init = 1 : i32 }
    %shim_lock_a_read  = aie.lock(%tile_0_0, 1) { init = 0 : i32 }
    %shim_lock_b_write = aie.lock(%tile_0_0, 2) { init = 1 : i32 }
    %shim_lock_b_read  = aie.lock(%tile_0_0, 3) { init = 0 : i32 }
    %shim_lock_c_write = aie.lock(%tile_0_0, 4) { init = 1 : i32 }
    %shim_lock_c_read  = aie.lock(%tile_0_0, 5) { init = 0 : i32 }

    %shim_mem = aie.shim_dma(%tile_0_0) {
      %0 = aie.dma_start(MM2S, 0, ^send_a, ^ch1)
    ^ch1:
      %1 = aie.dma_start(MM2S, 1, ^send_b, ^ch2)
    ^ch2:
      %2 = aie.dma_start(S2MM, 0, ^recv_c, ^done)
    ^send_a:
      aie.use_lock(%shim_lock_a_read, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%ext_a : memref<1024xi32> offset = 0 len = 1024) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%shim_lock_a_write, Release, %c1_i32)
      aie.next_bd ^send_a
    ^send_b:
      aie.use_lock(%shim_lock_b_read, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%ext_b : memref<1024xi32> offset = 0 len = 1024) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%shim_lock_b_write, Release, %c1_i32)
      aie.next_bd ^send_b
    ^recv_c:
      aie.use_lock(%shim_lock_c_write, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%ext_c : memref<1024xi32> offset = 0 len = 1024) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%shim_lock_c_read, Release, %c1_i32)
      aie.next_bd ^recv_c
    ^done:
      aie.end
    }

    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_0, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)
  }
}

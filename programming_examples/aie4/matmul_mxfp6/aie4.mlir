//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// MMAC MXFP6 (E2M3+E8M0) matrix multiply on a single core tile.
// C[4x64] += A[4x256] * B[256x64], K_outer=4, K_inner=64.
// MXFP6: block_size=32, 100 bytes per 128-element vector.
// Internally upconverted to MXFP8 by AIE4 hardware.
//
// RUN: xchesscc_wrapper aie4 -c %S/kernel.cc
// RUN: aiecc --alloc-scheme=basic-sequential --xchesscc --xbridge --get-txn --txn-name=transaction.bin --get-aiesim %s %test_utils_flags -- %S/test_sim.cpp
// RUN: aie4.mlir.prj/aiesim.sh | FileCheck --check-prefix=AIESIM %s
// XFAIL: *
//
// AIESIM: AIE4{{.*}}sim
// AIESIM: PASS!
//

module @matmul_mxfp6 {
  aie.device(npu3) {
    %c1_i32 = arith.constant 1 : i32
    %tile_0_2 = aie.tile(0, 2)

    // A: 4*2*100 = 800 bytes (4 K_outer iters, M/2=2 v128 per iter, 100 bytes each)
    %buf_a = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "buf_a"} : memref<800xi8>
    // B: 4*32*100 = 12800 bytes (4 K_outer iters, 32 v128 per iter)
    %buf_b = aie.buffer(%tile_0_2) {address = 32768 : i32, sym_name = "buf_b"} : memref<12800xi8>
    // C: 4*64 fp32 = 1024 bytes = 256 i32 words
    %buf_c = aie.buffer(%tile_0_2) {address = 65536 : i32, sym_name = "buf_c"} : memref<256xi32>

    %lock_input = aie.lock(%tile_0_2, 1) { init = 0 : i32 }
    %lock_output = aie.lock(%tile_0_2, 3) { init = 0 : i32 }

    func.func private @matmul(%A: memref<800xi8>, %B: memref<12800xi8>, %C: memref<256xi32>) -> ()

    %core02 = aie.core(%tile_0_2) {
      %true_init = arith.constant true
      scf.while (%cond = %true_init) : (i1) -> () {
        scf.condition(%cond)
      } do {
      ^bb0():
        aie.use_lock(%lock_input, AcquireGreaterEqual, %c1_i32)
        func.call @matmul(%buf_a, %buf_b, %buf_c) : (memref<800xi8>, memref<12800xi8>, memref<256xi32>) -> ()
        aie.use_lock(%lock_output, Release, %c1_i32)
        %true_next = arith.constant true
        scf.yield %true_next : i1
      }
      aie.end
    } { link_with="kernel.o" }
  }
}

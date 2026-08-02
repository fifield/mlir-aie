//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Elementwise fp16 * fp16 -> fp32 on a single core tile.
// 256 fp16 inputs A and B, 256 fp32 outputs C.
//
// RUN: xchesscc_wrapper aie4 -c %S/kernel.cc
// RUN: aiecc --alloc-scheme=basic-sequential --xchesscc --xbridge --get-txn --txn-name=transaction.bin --get-aiesim %s %test_utils_flags -- %S/test_sim.cpp
// RUN: aie4.mlir.prj/aiesim.sh | FileCheck --check-prefix=AIESIM %s
// XFAIL: *
//
// AIESIM: AIE4{{.*}}sim
// AIESIM: PASS!
//

module @vmul_fp16 {
  aie.device(npu3) {
    %c1_i32 = arith.constant 1 : i32
    %tile_0_2 = aie.tile(0, 2)

    // Input A: 256 x f16 = 512 bytes
    %buf_a = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "buf_a"} : memref<256xf16>
    // Input B: 256 x f16 = 512 bytes
    %buf_b = aie.buffer(%tile_0_2) {address = 1536 : i32, sym_name = "buf_b"} : memref<256xf16>
    // Output C: 256 x f32 = 1024 bytes
    %buf_c = aie.buffer(%tile_0_2) {address = 2048 : i32, sym_name = "buf_c"} : memref<256xf32>

    %lock_input = aie.lock(%tile_0_2, 1) { init = 0 : i32 }
    %lock_output = aie.lock(%tile_0_2, 3) { init = 0 : i32 }

    func.func private @vmul_fp16(%A: memref<256xf16>, %B: memref<256xf16>, %C: memref<256xf32>) -> ()

    %core02 = aie.core(%tile_0_2) {
      %true_init = arith.constant true
      scf.while (%cond = %true_init) : (i1) -> () {
        scf.condition(%cond)
      } do {
      ^bb0():
        aie.use_lock(%lock_input, AcquireGreaterEqual, %c1_i32)
        func.call @vmul_fp16(%buf_a, %buf_b, %buf_c) : (memref<256xf16>, memref<256xf16>, memref<256xf32>) -> ()
        aie.use_lock(%lock_output, Release, %c1_i32)
        %true_next = arith.constant true
        scf.yield %true_next : i1
      }
      aie.end
    } { link_with="kernel.o" }
  }
}

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
    %tile13 = aie.tile(1, 3)

    %buf13_0 = aie.buffer(%tile13) { sym_name = "a", address = 1024 : i32 } : memref<256xi32>
    %buf13_1 = aie.buffer(%tile13) { sym_name = "b", address = 32768 : i32 } : memref<4096xi32>
    %buf13_2 = aie.buffer(%tile13) { sym_name = "c", address = 65536 : i32 } : memref<256xi32>

    %lock13_1 = aie.lock(%tile13, 1) { sym_name = "input_lock", init = 0 : i32 }
    %lock13_3 = aie.lock(%tile13, 3) { sym_name = "output_lock", init = 0 : i32  }

    func.func private @matmul(%A: memref<256xi32>, %B: memref<4096xi32>, %C: memref<256xi32>) -> ()

    %core13 = aie.core(%tile13) {
        %true_init = arith.constant true
        scf.while (%cond = %true_init) : (i1) -> () {
           // Condition block
           scf.condition(%cond)
        } do {
          // Body block
          ^bb0():	      
          aie.use_lock(%lock13_1, AcquireGreaterEqual, 1)
          func.call @matmul(%buf13_0, %buf13_1, %buf13_2) : (memref<256xi32>, memref<4096xi32>, memref<256xi32>) -> ()
          aie.use_lock(%lock13_3, Release)
          // Loop back unconditionally
          %true_next = arith.constant true
          scf.yield %true_next : i1
       }
       aie.end
    } { link_with="kernel.o" }
  }
}

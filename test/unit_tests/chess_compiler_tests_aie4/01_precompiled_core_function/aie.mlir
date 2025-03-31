//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license, aietools_aie4, !hsa

// RUN: xchesscc_wrapper aie4 -c %S/kernel.cc
// RUN: %PYTHON aiecc.py -v --aiesim --xchesscc --xbridge --no-compile-host %s %test_utils_flags %S/test.cpp
// RUN: aie-opt --convert-aie-to-transaction='elf-dir=.' aie.mlir.prj/input_physical.mlir -o transaction.mlir
// RUN: aie-translate --aie-npu-to-binary transaction.mlir -o transaction.bin
// RUN: aie.mlir.prj/aiesim.sh | FileCheck %s

// CHECK: AIE4{{.*}}ISS
// CHECK: test start.
// CHECK: PASS!

module @test_chesss_01_precompiled_core_function {
  aie.device(npu3) {
    %tile13 = aie.tile(1, 3)

    %buf13_0 = aie.buffer(%tile13) { sym_name = "a", address = 1024 : i32 } : memref<256xi32>
    %buf13_1 = aie.buffer(%tile13) { sym_name = "b", address = 32768 : i32 } : memref<256xi32>

    %lock13_1 = aie.lock(%tile13, 1) { sym_name = "input_lock", init = 0 : i32 }
    %lock13_3 = aie.lock(%tile13, 3) { sym_name = "output_lock", init = 0 : i32  }

    func.func private @func(%A: memref<256xi32>, %B: memref<256xi32>) -> ()

    %core13 = aie.core(%tile13) {
      aie.use_lock(%lock13_1, AcquireGreaterEqual, 1)
      func.call @func(%buf13_0, %buf13_1) : (memref<256xi32>, memref<256xi32>) -> ()
      aie.use_lock(%lock13_3, Release)
      aie.end
    } { link_with="kernel.o" }
  }
}

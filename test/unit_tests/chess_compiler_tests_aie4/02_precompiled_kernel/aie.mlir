//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license, !hsa

// RUN: %PYTHON aiecc.py --aiesim --xchesscc --xbridge --dump-intermediates %s %test_utils_flags %S/test.cpp
// RUN: xchesscc_wrapper aie4 +l aie.mlir.prj/main_core_1_3.bcf %S/kernel.cc -o custom_1_3.elf
// RUN: aie-opt --convert-aie-to-transaction='elf-dir=.' aie.mlir.prj/input_physical.mlir -o transaction.mlir
// RUN: aie-translate --aie-npu-to-binary transaction.mlir -o transaction.bin
// RUN: aie.mlir.prj/aiesim.sh | FileCheck %s

// CHECK: AIE4{{.*}}sim
// CHECK: test start.
// CHECK: PASS!

module @test_chess_02_deprecated_precompiled_kernel {
  aie.device(npu3) {
    %tile13 = aie.tile(1, 3)

    %buf13_0 = aie.buffer(%tile13) { sym_name = "a", address = 1024 : i32 } : memref<256xi32>
    %buf13_1 = aie.buffer(%tile13) { sym_name = "b", address = 32768 : i32 } : memref<256xi32>

    %lock13_3 = aie.lock(%tile13, 3) { sym_name = "input_lock" }
    %lock13_5 = aie.lock(%tile13, 5) { sym_name = "output_lock" }

    %core13 = aie.core(%tile13) { aie.end } { elf_file = "custom_1_3.elf" }
  }
}

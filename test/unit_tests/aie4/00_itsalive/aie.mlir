//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

// CHECK: aie.device(npu3)

module @test00_itsalive {
  aie.device(npu3) {
    %tile22 = aie.tile(2, 2)

    %buf22_0 = aie.buffer(%tile22) { sym_name = "a", address = 0 : i32 } : memref<256xi32>

    %core22 = aie.core(%tile22) {
      %val1 = arith.constant 1 : i32
      %idx1 = arith.constant 3 : index
      %2 = arith.addi %val1, %val1 : i32
      aie.end
    }
  }
}
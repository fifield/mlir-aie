// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Translate each mode attribute to the trailing i32 intrinsic argument.
// Encodings from Peano's aie4_enums.h: 90 (4x4), 72 (8x8), 66/67 (zip lo/hi).

// RUN: aie-translate %s -mlir-to-llvmir -split-input-file | FileCheck %s

// CHECK-LABEL: define <16 x i32> @vshuffle_t32_4x4
llvm.func @vshuffle_t32_4x4(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // CHECK: call <16 x i32> @llvm.aie4.vshuffle(<16 x i32> %{{[0-9]+}}, <16 x i32> %{{[0-9]+}}, i32 90)
  %0 = "xllvm.intr.aie4.vshuffle"(%a, %b) <{mode = 90 : i32}> :
      (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
  llvm.return %0 : vector<16xi32>
}

// -----

// CHECK-LABEL: define <16 x i32> @vshuffle_t8_8x8
llvm.func @vshuffle_t8_8x8(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // CHECK: call <16 x i32> @llvm.aie4.vshuffle(<16 x i32> %{{[0-9]+}}, <16 x i32> %{{[0-9]+}}, i32 72)
  %0 = "xllvm.intr.aie4.vshuffle"(%a, %b) <{mode = 72 : i32}> :
      (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
  llvm.return %0 : vector<16xi32>
}

// -----

// CHECK-LABEL: define <16 x i32> @vshuffle_t32_2x16_lo
llvm.func @vshuffle_t32_2x16_lo(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // CHECK: call <16 x i32> @llvm.aie4.vshuffle(<16 x i32> %{{[0-9]+}}, <16 x i32> %{{[0-9]+}}, i32 66)
  %0 = "xllvm.intr.aie4.vshuffle"(%a, %b) <{mode = 66 : i32}> :
      (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
  llvm.return %0 : vector<16xi32>
}

// -----

// CHECK-LABEL: define <16 x i32> @vshuffle_t32_2x16_hi
llvm.func @vshuffle_t32_2x16_hi(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // CHECK: call <16 x i32> @llvm.aie4.vshuffle(<16 x i32> %{{[0-9]+}}, <16 x i32> %{{[0-9]+}}, i32 67)
  %0 = "xllvm.intr.aie4.vshuffle"(%a, %b) <{mode = 67 : i32}> :
      (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
  llvm.return %0 : vector<16xi32>
}

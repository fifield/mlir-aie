// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lower all four modes to the fixed vector<16xi32> intrinsic signature.
// Single-operand modes duplicate lhs; i8 and f32 vectors require bitcasts.

// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" | FileCheck %s

// CHECK-LABEL: @shuffle_t32_4x4
func.func @shuffle_t32_4x4(%a: vector<16xi32>) -> vector<16xi32> {
  // CHECK: "xllvm.intr.aie4.vshuffle"(%[[A:.*]], %[[A]]) <{mode = 90 : i32}> : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
  %0 = aievec.aie4.shuffle %a [mode = <t32_4x4>] : vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

// CHECK-LABEL: @shuffle_t8_8x8
func.func @shuffle_t8_8x8(%a: vector<64xi8>) -> vector<64xi8> {
  // CHECK: %[[B:.*]] = llvm.bitcast %{{.*}} : vector<64xi8> to vector<16xi32>
  // CHECK: "xllvm.intr.aie4.vshuffle"(%[[B]], %{{.*}}) <{mode = 72 : i32}> : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
  %0 = aievec.aie4.shuffle %a [mode = <t8_8x8>] : vector<64xi8>
  return %0 : vector<64xi8>
}

// -----

// CHECK-LABEL: @shuffle_t32_2x16_lo
func.func @shuffle_t32_2x16_lo(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // CHECK: "xllvm.intr.aie4.vshuffle"(%[[A:.*]], %[[B:.*]]) <{mode = 66 : i32}> : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
  %0 = aievec.aie4.shuffle %a, %b [mode = <t32_2x16_lo>] : vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

// CHECK-LABEL: @shuffle_t32_2x16_hi
func.func @shuffle_t32_2x16_hi(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // CHECK: "xllvm.intr.aie4.vshuffle"(%[[A:.*]], %[[B:.*]]) <{mode = 67 : i32}> : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
  %0 = aievec.aie4.shuffle %a, %b [mode = <t32_2x16_hi>] : vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

// CHECK-LABEL: @shuffle_t32_4x4_f32
func.func @shuffle_t32_4x4_f32(%a: vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[B:.*]] = llvm.bitcast %{{.*}} : vector<16xf32> to vector<16xi32>
  // CHECK: %[[R:.*]] = "xllvm.intr.aie4.vshuffle"(%[[B]], %{{.*}}) <{mode = 90 : i32}> : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
  // CHECK: llvm.bitcast %[[R]] : vector<16xi32> to vector<16xf32>
  %0 = aievec.aie4.shuffle %a [mode = <t32_4x4>] : vector<16xf32>
  return %0 : vector<16xf32>
}

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --lower-vector-to-aievec="aie-target=aie4b" | FileCheck %s --implicit-check-not=aievec.aie4.conv_f32_bf16
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" | FileCheck %s --implicit-check-not=aievec.aie4.conv_f32_bf16

// CHECK-LABEL: @scalar_and_native
// CHECK: arith.addf %arg0, %arg1 : f32
// CHECK: arith.subf %arg0, %arg1 : f32
// CHECK: aievec.aie4.mul_ewise_f32
func.func @scalar_and_native(%a: f32, %b: f32, %v: vector<32xf32>, %w: vector<32xf32>) -> (f32, f32, vector<32xf32>) {
  %a0 = arith.addf %a, %b : f32
  %s0 = arith.subf %a, %b : f32
  %m = arith.mulf %v, %w : vector<32xf32>
  return %a0, %s0, %m : f32, f32, vector<32xf32>
}

// CHECK-LABEL: @narrow_default
// CHECK: arith.truncf %arg0 : vector<32xf32> to vector<32xbf16>
func.func @narrow_default(%arg0: vector<32xf32>) -> vector<32xbf16> {
  %r = arith.truncf %arg0 : vector<32xf32> to vector<32xbf16>
  return %r : vector<32xbf16>
}

// CHECK-LABEL: @narrow_to_nearest_even
// CHECK: arith.truncf %arg0 to_nearest_even : vector<32xf32> to vector<32xbf16>
func.func @narrow_to_nearest_even(%arg0: vector<32xf32>) -> vector<32xbf16> {
  %r = arith.truncf %arg0 to_nearest_even : vector<32xf32> to vector<32xbf16>
  return %r : vector<32xbf16>
}

// CHECK-LABEL: @narrow_downward
// CHECK: arith.truncf %arg0 downward : vector<32xf32> to vector<32xbf16>
func.func @narrow_downward(%arg0: vector<32xf32>) -> vector<32xbf16> {
  %r = arith.truncf %arg0 downward : vector<32xf32> to vector<32xbf16>
  return %r : vector<32xbf16>
}

// CHECK-LABEL: @narrow_upward
// CHECK: arith.truncf %arg0 upward : vector<32xf32> to vector<32xbf16>
func.func @narrow_upward(%arg0: vector<32xf32>) -> vector<32xbf16> {
  %r = arith.truncf %arg0 upward : vector<32xf32> to vector<32xbf16>
  return %r : vector<32xbf16>
}

// CHECK-LABEL: @narrow_toward_zero
// CHECK: arith.truncf %arg0 toward_zero : vector<32xf32> to vector<32xbf16>
func.func @narrow_toward_zero(%arg0: vector<32xf32>) -> vector<32xbf16> {
  %r = arith.truncf %arg0 toward_zero : vector<32xf32> to vector<32xbf16>
  return %r : vector<32xbf16>
}

// CHECK-LABEL: @narrow_to_nearest_away
// CHECK: arith.truncf %arg0 to_nearest_away : vector<32xf32> to vector<32xbf16>
func.func @narrow_to_nearest_away(%arg0: vector<32xf32>) -> vector<32xbf16> {
  %r = arith.truncf %arg0 to_nearest_away : vector<32xf32> to vector<32xbf16>
  return %r : vector<32xbf16>
}

// CHECK-LABEL: @extf_scalable
// CHECK: arith.extf %arg0 : vector<[32]xbf16> to vector<[32]xf32>
func.func @extf_scalable(%arg0: vector<[32]xbf16>) -> vector<[32]xf32> {
  %r = arith.extf %arg0 : vector<[32]xbf16> to vector<[32]xf32>
  return %r : vector<[32]xf32>
}

// CHECK-LABEL: @extsi_scalable
// CHECK: arith.extsi %arg0 : vector<[32]xi16> to vector<[32]xi32>
func.func @extsi_scalable(%arg0: vector<[32]xi16>) -> vector<[32]xi32> {
  %r = arith.extsi %arg0 : vector<[32]xi16> to vector<[32]xi32>
  return %r : vector<[32]xi32>
}

// CHECK-LABEL: @extui_scalable
// CHECK: arith.extui %arg0 : vector<[32]xi16> to vector<[32]xi32>
func.func @extui_scalable(%arg0: vector<[32]xi16>) -> vector<[32]xi32> {
  %r = arith.extui %arg0 : vector<[32]xi16> to vector<[32]xi32>
  return %r : vector<[32]xi32>
}

// CHECK-LABEL: @extf_rank_two
// CHECK: arith.extf %arg0 : vector<4x8xbf16> to vector<4x8xf32>
func.func @extf_rank_two(%arg0: vector<4x8xbf16>) -> vector<4x8xf32> {
  %r = arith.extf %arg0 : vector<4x8xbf16> to vector<4x8xf32>
  return %r : vector<4x8xf32>
}

// CHECK-LABEL: @extsi_rank_two
// CHECK: arith.extsi %arg0 : vector<4x8xi16> to vector<4x8xi32>
func.func @extsi_rank_two(%arg0: vector<4x8xi16>) -> vector<4x8xi32> {
  %r = arith.extsi %arg0 : vector<4x8xi16> to vector<4x8xi32>
  return %r : vector<4x8xi32>
}

// CHECK-LABEL: @extui_rank_two
// CHECK: arith.extui %arg0 : vector<4x8xi16> to vector<4x8xi32>
func.func @extui_rank_two(%arg0: vector<4x8xi16>) -> vector<4x8xi32> {
  %r = arith.extui %arg0 : vector<4x8xi16> to vector<4x8xi32>
  return %r : vector<4x8xi32>
}

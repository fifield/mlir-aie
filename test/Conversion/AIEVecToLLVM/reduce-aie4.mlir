// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" | FileCheck %s

// Integer intrinsics require a seed; floating intrinsics have distinct
// seeded and unseeded forms.

// CHECK-LABEL: @reduce_add_i32_unseeded
func.func @reduce_add_i32_unseeded(%a: vector<32xi32>) -> i32 {
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: "xllvm.intr.aie4.add.reduce.I1024"(%{{.*}}, %[[ZERO]])
  %0 = aievec.aie4.reduce <add> %a : vector<32xi32> -> i32
  return %0 : i32
}

// CHECK-LABEL: @reduce_min_i32_unseeded
func.func @reduce_min_i32_unseeded(%a: vector<32xi32>) -> i32 {
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: "xllvm.intr.aie4.min.reduce.I1024"(%{{.*}}, %[[ZERO]])
  %0 = aievec.aie4.reduce <min> %a : vector<32xi32> -> i32
  return %0 : i32
}

// CHECK-LABEL: @reduce_max_i32_unseeded
func.func @reduce_max_i32_unseeded(%a: vector<32xi32>) -> i32 {
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: "xllvm.intr.aie4.max.reduce.I1024"(%{{.*}}, %[[ZERO]])
  %0 = aievec.aie4.reduce <max> %a : vector<32xi32> -> i32
  return %0 : i32
}

// -----

// Seeded integer reduction forwards its seed.

// CHECK-LABEL: @reduce_add_i32_seeded
func.func @reduce_add_i32_seeded(%a: vector<32xi32>, %seed: i32) -> i32 {
  // CHECK-NOT: llvm.mlir.constant
  // CHECK: "xllvm.intr.aie4.add.reduce.I1024"(%{{.*}}, %{{.*}})
  %0 = aievec.aie4.reduce <add> %a, %seed : vector<32xi32>, i32 -> i32
  return %0 : i32
}

// -----

// Unseeded floating reduction uses the one-operand intrinsic.

// CHECK-LABEL: @reduce_add_f32_unseeded
func.func @reduce_add_f32_unseeded(%a: vector<32xf32>) -> f32 {
  // CHECK: "xllvm.intr.aie4.add.reduce.FP1024"(%{{.*}}) : (vector<32xf32>) -> f32
  %0 = aievec.aie4.reduce <add> %a : vector<32xf32> -> f32
  return %0 : f32
}

// CHECK-LABEL: @reduce_min_f32_unseeded
func.func @reduce_min_f32_unseeded(%a: vector<32xf32>) -> f32 {
  // CHECK: "xllvm.intr.aie4.min.reduce.FP1024"(%{{.*}}) : (vector<32xf32>) -> f32
  %0 = aievec.aie4.reduce <min> %a : vector<32xf32> -> f32
  return %0 : f32
}

// CHECK-LABEL: @reduce_max_f32_unseeded
func.func @reduce_max_f32_unseeded(%a: vector<32xf32>) -> f32 {
  // CHECK: "xllvm.intr.aie4.max.reduce.FP1024"(%{{.*}}) : (vector<32xf32>) -> f32
  %0 = aievec.aie4.reduce <max> %a : vector<32xf32> -> f32
  return %0 : f32
}

// -----

// Seeded floating reduction uses the `.scl` intrinsic.

// CHECK-LABEL: @reduce_add_f32_seeded
func.func @reduce_add_f32_seeded(%a: vector<32xf32>, %seed: f32) -> f32 {
  // CHECK: "xllvm.intr.aie4.add.reduce.FP1024.scl"(%{{.*}}, %{{.*}}) : (vector<32xf32>, f32) -> f32
  %0 = aievec.aie4.reduce <add> %a, %seed : vector<32xf32>, f32 -> f32
  return %0 : f32
}

// CHECK-LABEL: @reduce_min_f32_seeded
func.func @reduce_min_f32_seeded(%a: vector<32xf32>, %seed: f32) -> f32 {
  // CHECK: "xllvm.intr.aie4.min.reduce.FP1024.scl"(%{{.*}}, %{{.*}}) : (vector<32xf32>, f32) -> f32
  %0 = aievec.aie4.reduce <min> %a, %seed : vector<32xf32>, f32 -> f32
  return %0 : f32
}

// CHECK-LABEL: @reduce_max_f32_seeded
func.func @reduce_max_f32_seeded(%a: vector<32xf32>, %seed: f32) -> f32 {
  // CHECK: "xllvm.intr.aie4.max.reduce.FP1024.scl"(%{{.*}}, %{{.*}}) : (vector<32xf32>, f32) -> f32
  %0 = aievec.aie4.reduce <max> %a, %seed : vector<32xf32>, f32 -> f32
  return %0 : f32
}

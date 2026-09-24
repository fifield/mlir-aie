// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie4b" --convert-vector-to-aievec="aie-target=aie4b" -split-input-file | FileCheck %s

// Native lowering preserves signed min/max and reassociable f32 add.
// Integer and ordered f32 add retain generic lowering.

// CHECK-LABEL: @reduce_add_i32
func.func @reduce_add_i32(%a: vector<32xi32>) -> i32 {
  // CHECK: vector.reduction <add>, %{{.*}} : vector<32xi32> into i32
  %0 = vector.reduction <add>, %a : vector<32xi32> into i32
  return %0 : i32
}

// CHECK-LABEL: @reduce_add_i32_seeded
func.func @reduce_add_i32_seeded(%a: vector<32xi32>, %seed: i32) -> i32 {
  // CHECK: vector.reduction <add>, %{{.*}}, %{{.*}} : vector<32xi32> into i32
  %0 = vector.reduction <add>, %a, %seed : vector<32xi32> into i32
  return %0 : i32
}

// CHECK-LABEL: @reduce_minsi_i32
func.func @reduce_minsi_i32(%a: vector<32xi32>) -> i32 {
  // CHECK: aievec.aie4.reduce <min> %{{.*}} : vector<32xi32> -> i32
  %0 = vector.reduction <minsi>, %a : vector<32xi32> into i32
  return %0 : i32
}

// CHECK-LABEL: @reduce_maxsi_i32
func.func @reduce_maxsi_i32(%a: vector<32xi32>) -> i32 {
  // CHECK: aievec.aie4.reduce <max> %{{.*}} : vector<32xi32> -> i32
  %0 = vector.reduction <maxsi>, %a : vector<32xi32> into i32
  return %0 : i32
}

// CHECK-LABEL: @reduce_add_f32
func.func @reduce_add_f32(%a: vector<32xf32>) -> f32 {
  // CHECK: vector.reduction <add>, %{{.*}} : vector<32xf32> into f32
  %0 = vector.reduction <add>, %a : vector<32xf32> into f32
  return %0 : f32
}

// CHECK-LABEL: @reduce_add_f32_seeded
func.func @reduce_add_f32_seeded(%a: vector<32xf32>, %seed: f32) -> f32 {
  // CHECK: vector.reduction <add>, %{{.*}}, %{{.*}} : vector<32xf32> into f32
  %0 = vector.reduction <add>, %a, %seed : vector<32xf32> into f32
  return %0 : f32
}

// CHECK-LABEL: @reduce_minimumf_f32
func.func @reduce_minimumf_f32(%a: vector<32xf32>) -> f32 {
  // CHECK: aievec.aie4.reduce <min> %{{.*}} : vector<32xf32> -> f32
  %0 = vector.reduction <minimumf>, %a : vector<32xf32> into f32
  return %0 : f32
}

// CHECK-LABEL: @reduce_maximumf_f32
func.func @reduce_maximumf_f32(%a: vector<32xf32>) -> f32 {
  // CHECK: aievec.aie4.reduce <max> %{{.*}} : vector<32xf32> -> f32
  %0 = vector.reduction <maximumf>, %a : vector<32xf32> into f32
  return %0 : f32
}

// CHECK-LABEL: @reduce_add_f32_reassoc
func.func @reduce_add_f32_reassoc(%a: vector<32xf32>) -> f32 {
  // CHECK: aievec.aie4.reduce <add> %{{.*}} : vector<32xf32> -> f32
  %0 = vector.reduction <add>, %a fastmath<reassoc> : vector<32xf32> into f32
  return %0 : f32
}

// CHECK-LABEL: @reduce_add_f32_reassoc_seeded
func.func @reduce_add_f32_reassoc_seeded(%a: vector<32xf32>, %seed: f32) -> f32 {
  // CHECK: aievec.aie4.reduce <add> %{{.*}}, %{{.*}} : vector<32xf32>, f32 -> f32
  %0 = vector.reduction <add>, %a, %seed fastmath<reassoc> : vector<32xf32> into f32
  return %0 : f32
}

// CHECK-LABEL: @reduce_add_i32_scalable
func.func @reduce_add_i32_scalable(%a: vector<[32]xi32>) -> i32 {
  // CHECK: vector.reduction <add>, %{{.*}} : vector<[32]xi32> into i32
  %0 = vector.reduction <add>, %a : vector<[32]xi32> into i32
  return %0 : i32
}

// CHECK-LABEL: @reduce_add_f32_scalable_reassoc
func.func @reduce_add_f32_scalable_reassoc(%a: vector<[32]xf32>) -> f32 {
  // CHECK: vector.reduction <add>, %{{.*}} fastmath<reassoc> : vector<[32]xf32> into f32
  %0 = vector.reduction <add>, %a fastmath<reassoc> : vector<[32]xf32> into f32
  return %0 : f32
}

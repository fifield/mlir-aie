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

// -----

// AIE4 has no bf16 reduction. minimum/maximum widen to f32 with the native
// conversion, reduce there, and narrow the result, which is one of the
// inputs, back exactly.
// CHECK-LABEL: @reduce_maximumf_bf16
// CHECK-SAME: (%[[A:.*]]: vector<32xbf16>)
// CHECK: %[[W:.*]] = aievec.aie4.conv_bf16_f32 %[[A]]
// CHECK: %[[R:.*]] = aievec.aie4.reduce <max> %[[W]] : vector<32xf32> -> f32
// CHECK: %[[N:.*]] = arith.truncf %[[R]] : f32 to bf16
// CHECK: return %[[N]]
func.func @reduce_maximumf_bf16(%a: vector<32xbf16>) -> bf16 {
  %0 = vector.reduction <maximumf>, %a : vector<32xbf16> into bf16
  return %0 : bf16
}

// -----

// The accumulator widens with it.
// CHECK-LABEL: @reduce_minimumf_bf16_acc
// CHECK-SAME: (%[[A:.*]]: vector<32xbf16>, %[[S:.*]]: bf16)
// CHECK-DAG: %[[W:.*]] = aievec.aie4.conv_bf16_f32 %[[A]]
// CHECK-DAG: %[[WS:.*]] = arith.extf %[[S]] : bf16 to f32
// CHECK: %[[R:.*]] = aievec.aie4.reduce <min> %[[W]], %[[WS]] : vector<32xf32>, f32 -> f32
// CHECK: arith.truncf %[[R]] : f32 to bf16
func.func @reduce_minimumf_bf16_acc(%a: vector<32xbf16>, %s: bf16) -> bf16 {
  %0 = vector.reduction <minimumf>, %a, %s : vector<32xbf16> into bf16
  return %0 : bf16
}

// -----

// A bf16 add reduction would round, so it is not widened and stays generic.
// CHECK-LABEL: @reduce_add_bf16_unclaimed
// CHECK: vector.reduction <add>, %{{.*}} : vector<32xbf16> into bf16
// CHECK-NOT: aievec.aie4.reduce
func.func @reduce_add_bf16_unclaimed(%a: vector<32xbf16>) -> bf16 {
  %0 = vector.reduction <add>, %a : vector<32xbf16> into bf16
  return %0 : bf16
}

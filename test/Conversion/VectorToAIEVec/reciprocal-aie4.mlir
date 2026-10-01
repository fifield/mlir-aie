// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie4b" --convert-vector-to-aievec="aie-target=aie4b" | FileCheck %s

// Only fastmath<afn> scalar f32 1.0 / x uses the approximate INV instruction.
// Other divisions retain their existing lowering.

// CHECK-LABEL: @recip_afn
// CHECK-SAME: (%[[X:.*]]: f32)
// CHECK-NEXT: %[[R:.*]] = aievec.aie4.inv %[[X]] : f32
// CHECK-NEXT: return %[[R]]
func.func @recip_afn(%x: f32) -> f32 {
  %one = arith.constant 1.0 : f32
  %0 = arith.divf %one, %x fastmath<afn> : f32
  return %0 : f32
}

// CHECK-LABEL: @recip_fast
// CHECK: aievec.aie4.inv
func.func @recip_fast(%x: f32) -> f32 {
  %one = arith.constant 1.0 : f32
  %0 = arith.divf %one, %x fastmath<fast> : f32
  return %0 : f32
}

// CHECK-LABEL: @recip_strict
// CHECK: arith.divf %{{.*}}, %{{.*}} : f32
// CHECK-NOT: aievec.aie4.inv
func.func @recip_strict(%x: f32) -> f32 {
  %one = arith.constant 1.0 : f32
  %0 = arith.divf %one, %x : f32
  return %0 : f32
}

// arcp alone does not grant approximation here.
// CHECK-LABEL: @recip_arcp
// CHECK: arith.divf %{{.*}}, %{{.*}} fastmath<arcp,contract> : f32
// CHECK-NOT: aievec.aie4.inv
func.func @recip_arcp(%x: f32) -> f32 {
  %one = arith.constant 1.0 : f32
  %0 = arith.divf %one, %x fastmath<arcp,contract> : f32
  return %0 : f32
}

// CHECK-LABEL: @div_afn_not_one
// CHECK: arith.divf %{{.*}}, %{{.*}} fastmath<afn> : f32
// CHECK-NOT: aievec.aie4.inv
func.func @div_afn_not_one(%x: f32) -> f32 {
  %two = arith.constant 2.0 : f32
  %0 = arith.divf %two, %x fastmath<afn> : f32
  return %0 : f32
}

// CHECK-LABEL: @div_afn_variable
// CHECK: arith.divf %{{.*}}, %{{.*}} fastmath<afn> : f32
func.func @div_afn_variable(%x: f32, %y: f32) -> f32 {
  %0 = arith.divf %y, %x fastmath<afn> : f32
  return %0 : f32
}

// CHECK-LABEL: @recip_afn_vector
// CHECK: arith.divf %{{.*}}, %{{.*}} fastmath<afn> : vector<32xf32>
// CHECK-NOT: aievec
func.func @recip_afn_vector(%x: vector<32xf32>) -> vector<32xf32> {
  %one = arith.constant dense<1.0> : vector<32xf32>
  %0 = arith.divf %one, %x fastmath<afn> : vector<32xf32>
  return %0 : vector<32xf32>
}

// CHECK-LABEL: @recip_afn_bf16
// CHECK: arith.divf %{{.*}}, %{{.*}} fastmath<afn> : bf16
func.func @recip_afn_bf16(%x: bf16) -> bf16 {
  %one = arith.constant 1.0 : bf16
  %0 = arith.divf %one, %x fastmath<afn> : bf16
  return %0 : bf16
}

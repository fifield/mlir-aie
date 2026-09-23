// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --lower-vector-to-aievec="aie-target=aie4b" | FileCheck %s --implicit-check-not=arith.mulf

// A returned product must retain a native multiply even when another use fuses.
// CHECK-LABEL: @returned_product
// CHECK: %[[M:.*]] = aievec.aie4.mul_ewise_f32
// CHECK: return %[[M]],
func.func @returned_product(%a: vector<32xf32>, %b: vector<32xf32>, %acc: vector<32xf32>) -> (vector<32xf32>, vector<32xf32>) {
  %m = arith.mulf %a, %b fastmath<contract> : vector<32xf32>
  %s = arith.addf %acc, %m fastmath<contract> : vector<32xf32>
  return %m, %s : vector<32xf32>, vector<32xf32>
}

// Only the RHS product is consumed; the LHS becomes the accumulator.
// CHECK-LABEL: @two_products
// CHECK: %[[M:.*]] = aievec.aie4.mul_ewise_f32 %arg0, %arg1
// CHECK: aievec.aie4.mac_ewise_f32 %arg2, %arg3, %[[M]]
func.func @two_products(%a: vector<32xf32>, %b: vector<32xf32>, %c: vector<32xf32>, %d: vector<32xf32>) -> vector<32xf32> {
  %m = arith.mulf %a, %b fastmath<contract> : vector<32xf32>
  %n = arith.mulf %c, %d fastmath<contract> : vector<32xf32>
  %s = arith.addf %m, %n fastmath<contract> : vector<32xf32>
  return %s : vector<32xf32>
}

// A single user can still need the product as its accumulator.
// CHECK-LABEL: @product_as_accumulator
// CHECK: aievec.aie4.mul_ewise_f32
// CHECK: return
func.func @product_as_accumulator(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<32xf32> {
  %m = arith.mulf %a, %b fastmath<contract> : vector<32xf32>
  %s = arith.addf %m, %m fastmath<contract> : vector<32xf32>
  return %s : vector<32xf32>
}

// Multiple uses may be deferred when every use disappears through fusion.
// CHECK-LABEL: @two_fused_users
// CHECK-NOT: aievec.aie4.mul_ewise
// CHECK: aievec.aie4.mac_ewise_f32
// CHECK-NEXT: {{.*}}aievec.aie4.mac_ewise_f32
// CHECK-NEXT: return
func.func @two_fused_users(%a: vector<32xf32>, %b: vector<32xf32>, %acc0: vector<32xf32>, %acc1: vector<32xf32>) -> (vector<32xf32>, vector<32xf32>) {
  %m = arith.mulf %a, %b fastmath<contract> : vector<32xf32>
  %s0 = arith.addf %acc0, %m fastmath<contract> : vector<32xf32>
  %s1 = arith.addf %acc1, %m fastmath<contract> : vector<32xf32>
  return %s0, %s1 : vector<32xf32>, vector<32xf32>
}

// An add without contraction permission must still use a native product.
// CHECK-LABEL: @unfused_user
// CHECK: %[[M:.*]] = aievec.aie4.mul_ewise_f32
// CHECK: %[[S:.*]] = arith.addf %arg3, %[[M]] :
// CHECK: return {{.*}}, %[[S]]
func.func @unfused_user(%a: vector<32xf32>, %b: vector<32xf32>, %acc0: vector<32xf32>, %acc1: vector<32xf32>) -> (vector<32xf32>, vector<32xf32>) {
  %m = arith.mulf %a, %b fastmath<contract> : vector<32xf32>
  %s0 = arith.addf %acc0, %m fastmath<contract> : vector<32xf32>
  %s1 = arith.addf %acc1, %m : vector<32xf32>
  return %s0, %s1 : vector<32xf32>, vector<32xf32>
}

// CHECK-LABEL: @bf16_returned_product
// CHECK: %[[M:.*]] = aievec.aie4.mul_ewise_bf16_f32
// CHECK: return %[[M]],
func.func @bf16_returned_product(%a: vector<32xbf16>, %b: vector<32xbf16>, %acc: vector<32xf32>) -> (vector<32xf32>, vector<32xf32>) {
  %ea = arith.extf %a : vector<32xbf16> to vector<32xf32>
  %eb = arith.extf %b : vector<32xbf16> to vector<32xf32>
  %m = arith.mulf %ea, %eb fastmath<contract> : vector<32xf32>
  %s = arith.addf %acc, %m fastmath<contract> : vector<32xf32>
  return %m, %s : vector<32xf32>, vector<32xf32>
}

// CHECK-LABEL: @bf16_two_products
// CHECK: %[[M:.*]] = aievec.aie4.mul_ewise_bf16_f32 %arg0, %arg1
// CHECK: aievec.aie4.mac_ewise_bf16_f32 %arg2, %arg3, %[[M]]
func.func @bf16_two_products(%a: vector<32xbf16>, %b: vector<32xbf16>, %c: vector<32xbf16>, %d: vector<32xbf16>) -> vector<32xf32> {
  %ea = arith.extf %a : vector<32xbf16> to vector<32xf32>
  %eb = arith.extf %b : vector<32xbf16> to vector<32xf32>
  %ec = arith.extf %c : vector<32xbf16> to vector<32xf32>
  %ed = arith.extf %d : vector<32xbf16> to vector<32xf32>
  %m = arith.mulf %ea, %eb fastmath<contract> : vector<32xf32>
  %n = arith.mulf %ec, %ed fastmath<contract> : vector<32xf32>
  %s = arith.addf %m, %n fastmath<contract> : vector<32xf32>
  return %s : vector<32xf32>
}

// Both MAC rewrites must agree on which product to consume.
// CHECK-LABEL: @mixed_products
// CHECK: %[[M:.*]] = aievec.aie4.mul_ewise_f32 %arg0, %arg1
// CHECK: aievec.aie4.mac_ewise_bf16_f32 %arg2, %arg3, %[[M]]
func.func @mixed_products(%a: vector<32xf32>, %b: vector<32xf32>, %c: vector<32xbf16>, %d: vector<32xbf16>) -> vector<32xf32> {
  %ec = arith.extf %c : vector<32xbf16> to vector<32xf32>
  %ed = arith.extf %d : vector<32xbf16> to vector<32xf32>
  %m = arith.mulf %a, %b fastmath<contract> : vector<32xf32>
  %n = arith.mulf %ec, %ed fastmath<contract> : vector<32xf32>
  %s = arith.addf %m, %n fastmath<contract> : vector<32xf32>
  return %s : vector<32xf32>
}

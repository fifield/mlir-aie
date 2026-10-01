// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" -split-input-file | FileCheck %s

// CHECK-LABEL: @msc_f32
// CHECK: aievec.aie4.msc_ewise_f32
// CHECK-NOT: arith.subf
func.func @msc_f32(%a: vector<32xf32>, %b: vector<32xf32>,
                   %c: vector<32xf32>) -> vector<32xf32> {
  %m = arith.mulf %a, %b fastmath<contract> : vector<32xf32>
  %s = arith.subf %c, %m fastmath<contract> : vector<32xf32>
  return %s : vector<32xf32>
}

// -----

// CHECK-LABEL: @msc_bf16
// CHECK: aievec.aie4.msc_ewise_bf16_f32
// CHECK-NOT: arith.subf
func.func @msc_bf16(%a: vector<32xbf16>, %b: vector<32xbf16>,
                    %c: vector<32xf32>) -> vector<32xf32> {
  %ea = arith.extf %a : vector<32xbf16> to vector<32xf32>
  %eb = arith.extf %b : vector<32xbf16> to vector<32xf32>
  %m = arith.mulf %ea, %eb fastmath<contract> : vector<32xf32>
  %s = arith.subf %c, %m fastmath<contract> : vector<32xf32>
  return %s : vector<32xf32>
}

// -----

// Subtraction does not commute: `%a * %b - %acc` is not a multiply-subtract,
// so the product lowers on its own and the subtract stays.
// CHECK-LABEL: @product_on_the_left
// CHECK: aievec.aie4.mul_ewise_f32
// CHECK: arith.subf
// CHECK-NOT: aievec.aie4.msc_ewise_f32
func.func @product_on_the_left(%a: vector<32xf32>, %b: vector<32xf32>,
                               %c: vector<32xf32>) -> vector<32xf32> {
  %m = arith.mulf %a, %b fastmath<contract> : vector<32xf32>
  %s = arith.subf %m, %c fastmath<contract> : vector<32xf32>
  return %s : vector<32xf32>
}

// -----

// Fusion removes the product's rounding, so it needs contract on both ops.
// CHECK-LABEL: @msc_no_contract
// CHECK: arith.subf
// CHECK-NOT: aievec.aie4.msc_ewise_f32
func.func @msc_no_contract(%a: vector<32xf32>, %b: vector<32xf32>,
                           %c: vector<32xf32>) -> vector<32xf32> {
  %m = arith.mulf %a, %b : vector<32xf32>
  %s = arith.subf %c, %m : vector<32xf32>
  return %s : vector<32xf32>
}

// -----

// CHECK-LABEL: @msc_contract_on_sub_only
// CHECK: arith.subf
// CHECK-NOT: aievec.aie4.msc_ewise_f32
func.func @msc_contract_on_sub_only(%a: vector<32xf32>, %b: vector<32xf32>,
                                    %c: vector<32xf32>) -> vector<32xf32> {
  %m = arith.mulf %a, %b : vector<32xf32>
  %s = arith.subf %c, %m fastmath<contract> : vector<32xf32>
  return %s : vector<32xf32>
}

// -----

// The opposite one-sided case: contract on the multiply, absent on the
// subtract. Without this, dropping the subtract-side check would still pass.
// CHECK-LABEL: @msc_contract_on_mul_only
// CHECK: arith.subf
// CHECK-NOT: aievec.aie4.msc_ewise_f32
func.func @msc_contract_on_mul_only(%a: vector<32xf32>, %b: vector<32xf32>,
                                    %c: vector<32xf32>) -> vector<32xf32> {
  %m = arith.mulf %a, %b fastmath<contract> : vector<32xf32>
  %s = arith.subf %c, %m : vector<32xf32>
  return %s : vector<32xf32>
}

// -----

// Same one-sided case on the widened-bf16 path.
// CHECK-LABEL: @msc_bf16_contract_on_mul_only
// CHECK: arith.subf
// CHECK-NOT: aievec.aie4.msc_ewise_bf16_f32
func.func @msc_bf16_contract_on_mul_only(%a: vector<32xbf16>, %b: vector<32xbf16>,
                                         %c: vector<32xf32>) -> vector<32xf32> {
  %ea = arith.extf %a : vector<32xbf16> to vector<32xf32>
  %eb = arith.extf %b : vector<32xbf16> to vector<32xf32>
  %m = arith.mulf %ea, %eb fastmath<contract> : vector<32xf32>
  %s = arith.subf %c, %m : vector<32xf32>
  return %s : vector<32xf32>
}

// -----

// Contract on the subtract but not the multiply, on the bf16 path.
// CHECK-LABEL: @msc_bf16_contract_on_sub_only
// CHECK: arith.subf
// CHECK-NOT: aievec.aie4.msc_ewise_bf16_f32
func.func @msc_bf16_contract_on_sub_only(%a: vector<32xbf16>, %b: vector<32xbf16>,
                                         %c: vector<32xf32>) -> vector<32xf32> {
  %ea = arith.extf %a : vector<32xbf16> to vector<32xf32>
  %eb = arith.extf %b : vector<32xbf16> to vector<32xf32>
  %m = arith.mulf %ea, %eb : vector<32xf32>
  %s = arith.subf %c, %m fastmath<contract> : vector<32xf32>
  return %s : vector<32xf32>
}

// -----

// A width the MSC form does not cover stays legal.
// CHECK-LABEL: @msc_wrong_width
// CHECK: arith.subf {{.*}} : vector<16xf32>
func.func @msc_wrong_width(%a: vector<16xf32>, %b: vector<16xf32>,
                           %c: vector<16xf32>) -> vector<16xf32> {
  %m = arith.mulf %a, %b fastmath<contract> : vector<16xf32>
  %s = arith.subf %c, %m fastmath<contract> : vector<16xf32>
  return %s : vector<16xf32>
}

// -----

// The product feeds both an MSC root and another use, so it cannot be
// deferred: it lowers on its own and the subtract still fuses, which is how
// the MAC path behaves in the same situation.
// CHECK-LABEL: @product_escapes
// CHECK: aievec.aie4.mul_ewise_f32
// CHECK: aievec.aie4.msc_ewise_f32
func.func @product_escapes(%a: vector<32xf32>, %b: vector<32xf32>,
                           %c: vector<32xf32>) -> (vector<32xf32>, vector<32xf32>) {
  %m = arith.mulf %a, %b fastmath<contract> : vector<32xf32>
  %s = arith.subf %c, %m fastmath<contract> : vector<32xf32>
  return %s, %m : vector<32xf32>, vector<32xf32>
}

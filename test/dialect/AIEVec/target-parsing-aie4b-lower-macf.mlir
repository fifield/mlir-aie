//
// target-parsing-aie4b-lower-macf.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Float MAC fusion requires contract/fast on both mulf and addf.
// Widened bf16 multiplication alone needs no contraction permission.
//
// RUN: aie-opt %s -split-input-file --lower-vector-to-aievec="aie-target=aie4b" | FileCheck %s

// Contract on both operations permits fusion.
// CHECK-LABEL: func.func @f32_mac_contract_fuses
// CHECK: aievec.aie4.mac_ewise_f32 %arg0, %arg1, %arg2 : vector<32xf32>
// CHECK-NOT: arith.mulf
// CHECK-NOT: arith.addf
func.func @f32_mac_contract_fuses(%a : vector<32xf32>, %b : vector<32xf32>,
                                  %acc : vector<32xf32>) -> vector<32xf32> {
  %m = arith.mulf %a, %b fastmath<contract> : vector<32xf32>
  %s = arith.addf %acc, %m fastmath<contract> : vector<32xf32>
  return %s : vector<32xf32>
}

// -----

// `arith.addf` is commutative, so the product may appear on either side.
// CHECK-LABEL: func.func @f32_mac_operands_swapped
// CHECK: aievec.aie4.mac_ewise_f32 %arg0, %arg1, %arg2 : vector<32xf32>
func.func @f32_mac_operands_swapped(%a : vector<32xf32>, %b : vector<32xf32>,
                                    %acc : vector<32xf32>) -> vector<32xf32> {
  %m = arith.mulf %a, %b fastmath<contract> : vector<32xf32>
  %s = arith.addf %m, %acc fastmath<contract> : vector<32xf32>
  return %s : vector<32xf32>
}

// -----

// Without contract, retain a native multiply and a separate add.
// CHECK-LABEL: func.func @f32_mac_no_contract_does_not_fuse
// CHECK: %[[M:.*]] = aievec.aie4.mul_ewise_f32 %arg0, %arg1 : vector<32xf32>
// CHECK: arith.addf %arg2, %[[M]] : vector<32xf32>
// CHECK-NOT: aievec.aie4.mac_ewise_f32
func.func @f32_mac_no_contract_does_not_fuse(
    %a : vector<32xf32>, %b : vector<32xf32>, %acc : vector<32xf32>)
    -> vector<32xf32> {
  %m = arith.mulf %a, %b : vector<32xf32>
  %s = arith.addf %acc, %m : vector<32xf32>
  return %s : vector<32xf32>
}

// -----

// Contract on the add alone does not permit fusion.
// CHECK-LABEL: func.func @f32_mac_contract_on_add_only_does_not_fuse
// CHECK: aievec.aie4.mul_ewise_f32
// CHECK: arith.addf
// CHECK-NOT: aievec.aie4.mac_ewise_f32
func.func @f32_mac_contract_on_add_only_does_not_fuse(
    %a : vector<32xf32>, %b : vector<32xf32>, %acc : vector<32xf32>)
    -> vector<32xf32> {
  %m = arith.mulf %a, %b : vector<32xf32>
  %s = arith.addf %acc, %m fastmath<contract> : vector<32xf32>
  return %s : vector<32xf32>
}

// -----

// Contract on the multiply alone does not permit fusion.
// CHECK-LABEL: func.func @f32_mac_contract_on_mul_only_does_not_fuse
// CHECK: aievec.aie4.mul_ewise_f32
// CHECK: arith.addf
// CHECK-NOT: aievec.aie4.mac_ewise_f32
func.func @f32_mac_contract_on_mul_only_does_not_fuse(
    %a : vector<32xf32>, %b : vector<32xf32>, %acc : vector<32xf32>)
    -> vector<32xf32> {
  %m = arith.mulf %a, %b fastmath<contract> : vector<32xf32>
  %s = arith.addf %acc, %m : vector<32xf32>
  return %s : vector<32xf32>
}

// -----

// A plain add remains eligible for generic vadd.f lowering.
// CHECK-LABEL: func.func @f32_plain_addf_untouched
// CHECK: arith.addf
// CHECK-NOT: aievec.aie4.mac_ewise_f32
func.func @f32_plain_addf_untouched(%a : vector<32xf32>, %b : vector<32xf32>)
    -> vector<32xf32> {
  %0 = arith.addf %a, %b fastmath<contract> : vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

//===----------------------------------------------------------------------===//
// bf16 x bf16 -> f32 MUL/MAC
//===----------------------------------------------------------------------===//

// Native bf16 multiplication absorbs both widenings without contraction.
// CHECK-LABEL: func.func @bf16_mul_widened_no_contract_needed
// CHECK: aievec.aie4.mul_ewise_bf16_f32 %arg0, %arg1 : vector<32xbf16>, vector<32xf32>
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.mulf
func.func @bf16_mul_widened_no_contract_needed(
    %a : vector<32xbf16>, %b : vector<32xbf16>) -> vector<32xf32> {
  %ea = arith.extf %a : vector<32xbf16> to vector<32xf32>
  %eb = arith.extf %b : vector<32xbf16> to vector<32xf32>
  %m = arith.mulf %ea, %eb : vector<32xf32>
  return %m : vector<32xf32>
}

// -----

// Contract permits fusion of the widened bf16 product and add.
// CHECK-LABEL: func.func @bf16_mac_contract_fuses
// CHECK: aievec.aie4.mac_ewise_bf16_f32 %arg0, %arg1, %arg2 : vector<32xbf16>, vector<32xf32>
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.mulf
// CHECK-NOT: arith.addf
func.func @bf16_mac_contract_fuses(%a : vector<32xbf16>, %b : vector<32xbf16>,
                                   %acc : vector<32xf32>) -> vector<32xf32> {
  %ea = arith.extf %a : vector<32xbf16> to vector<32xf32>
  %eb = arith.extf %b : vector<32xbf16> to vector<32xf32>
  %m = arith.mulf %ea, %eb fastmath<contract> : vector<32xf32>
  %s = arith.addf %acc, %m fastmath<contract> : vector<32xf32>
  return %s : vector<32xf32>
}

// -----

// `arith.addf` is commutative here too.
// CHECK-LABEL: func.func @bf16_mac_operands_swapped
// CHECK: aievec.aie4.mac_ewise_bf16_f32 %arg0, %arg1, %arg2 : vector<32xbf16>, vector<32xf32>
func.func @bf16_mac_operands_swapped(
    %a : vector<32xbf16>, %b : vector<32xbf16>, %acc : vector<32xf32>)
    -> vector<32xf32> {
  %ea = arith.extf %a : vector<32xbf16> to vector<32xf32>
  %eb = arith.extf %b : vector<32xbf16> to vector<32xf32>
  %m = arith.mulf %ea, %eb fastmath<contract> : vector<32xf32>
  %s = arith.addf %m, %acc fastmath<contract> : vector<32xf32>
  return %s : vector<32xf32>
}

// -----

// Without contract, the bf16 multiply lowers but the add stays separate.
// CHECK-LABEL: func.func @bf16_mac_no_contract_does_not_fuse
// CHECK: %[[M:.*]] = aievec.aie4.mul_ewise_bf16_f32 %arg0, %arg1 : vector<32xbf16>, vector<32xf32>
// CHECK: arith.addf %arg2, %[[M]] : vector<32xf32>
// CHECK-NOT: aievec.aie4.mac_ewise_bf16_f32
func.func @bf16_mac_no_contract_does_not_fuse(
    %a : vector<32xbf16>, %b : vector<32xbf16>, %acc : vector<32xf32>)
    -> vector<32xf32> {
  %ea = arith.extf %a : vector<32xbf16> to vector<32xf32>
  %eb = arith.extf %b : vector<32xbf16> to vector<32xf32>
  %m = arith.mulf %ea, %eb : vector<32xf32>
  %s = arith.addf %acc, %m : vector<32xf32>
  return %s : vector<32xf32>
}

// -----

// Adding two widened bf16 values must remain a plain f32 add.
// CHECK-LABEL: func.func @f32_add_of_two_bf16_widenings_is_not_a_mac
// CHECK: %[[A:.*]] = aievec.aie4.conv_bf16_f32 %arg0
// CHECK: %[[B:.*]] = aievec.aie4.conv_bf16_f32 %arg1
// CHECK: arith.addf %[[A]], %[[B]]
// CHECK-NOT: aievec.aie4.mac_ewise_f32
// CHECK-NOT: aievec.aie4.mac_ewise_bf16_f32
func.func @f32_add_of_two_bf16_widenings_is_not_a_mac(
    %a : vector<32xbf16>, %b : vector<32xbf16>) -> vector<32xf32> {
  %ea = arith.extf %a : vector<32xbf16> to vector<32xf32>
  %eb = arith.extf %b : vector<32xbf16> to vector<32xf32>
  %s = arith.addf %ea, %eb fastmath<contract> : vector<32xf32>
  return %s : vector<32xf32>
}

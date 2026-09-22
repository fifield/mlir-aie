// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie4b" --convert-vector-to-aievec="aie-target=aie4b" -split-input-file | FileCheck %s

// Native integer min/max, comparisons, and selects support five fixed
// rank-one shapes. Comparisons and selects pack or unpack Boolean lanes;
// unsupported shapes and equality predicates use generic lowering.

// -- MIN/MAX --

// CHECK-LABEL: @minsi_i8x64
func.func @minsi_i8x64(%a: vector<64xi8>, %b: vector<64xi8>) -> vector<64xi8> {
  // CHECK: aievec.aie4.minmax %{{.*}}, %{{.*}}[kind = <min>, signed = true] : vector<64xi8>
  %0 = arith.minsi %a, %b : vector<64xi8>
  return %0 : vector<64xi8>
}

// CHECK-LABEL: @maxui_i32x16
func.func @maxui_i32x16(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // CHECK: aievec.aie4.minmax %{{.*}}, %{{.*}}[kind = <max>, signed = false] : vector<16xi32>
  %0 = arith.maxui %a, %b : vector<16xi32>
  return %0 : vector<16xi32>
}

// CHECK-LABEL: @minui_i16x64
func.func @minui_i16x64(%a: vector<64xi16>, %b: vector<64xi16>) -> vector<64xi16> {
  // CHECK: aievec.aie4.minmax %{{.*}}, %{{.*}}[kind = <min>, signed = false] : vector<64xi16>
  %0 = arith.minui %a, %b : vector<64xi16>
  return %0 : vector<64xi16>
}

// Unsupported shapes retain generic min/max.
// CHECK-LABEL: @minsi_out_of_scope
func.func @minsi_out_of_scope(%a: vector<8xi32>, %b: vector<8xi32>) -> vector<8xi32> {
  // CHECK: arith.minsi
  // CHECK-NOT: aievec.aie4.minmax
  %0 = arith.minsi %a, %b : vector<8xi32>
  return %0 : vector<8xi32>
}

// -----

// -- CMP --

// CHECK-LABEL: @cmpi_slt_i16x32
func.func @cmpi_slt_i16x32(%a: vector<32xi16>, %b: vector<32xi16>) -> vector<32xi1> {
  // CHECK: %[[C:.*]] = aievec.aie4.cmp %{{.*}}, %{{.*}}[pred = <lt>, signed = true] : vector<32xi16>, ui32
  // CHECK: %[[S:.*]] = builtin.unrealized_conversion_cast %[[C]] : ui32 to i32
  // CHECK: %[[V:.*]] = vector.broadcast %[[S]] : i32 to vector<1xi32>
  // CHECK: vector.bitcast %[[V]] : vector<1xi32> to vector<32xi1>
  %0 = arith.cmpi slt, %a, %b : vector<32xi16>
  return %0 : vector<32xi1>
}

// gt(a, b) lowers to lt(b, a).
// CHECK-LABEL: @cmpi_sgt_i16x32
// CHECK-SAME: %[[A:.*]]: vector<32xi16>, %[[B:.*]]: vector<32xi16>
func.func @cmpi_sgt_i16x32(%a: vector<32xi16>, %b: vector<32xi16>) -> vector<32xi1> {
  // CHECK: aievec.aie4.cmp %[[B]], %[[A]][pred = <lt>, signed = true] : vector<32xi16>, ui32
  %0 = arith.cmpi sgt, %a, %b : vector<32xi16>
  return %0 : vector<32xi1>
}

// Unsigned le(a, b) lowers to ge(b, a).
// CHECK-LABEL: @cmpi_ule_i32x16
// CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[B:.*]]: vector<16xi32>
func.func @cmpi_ule_i32x16(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi1> {
  // CHECK: aievec.aie4.cmp %[[B]], %[[A]][pred = <ge>, signed = false] : vector<16xi32>, ui32
  %0 = arith.cmpi ule, %a, %b : vector<16xi32>
  return %0 : vector<16xi1>
}

// Equality predicates retain generic comparison.
// CHECK-LABEL: @cmpi_eq_i16x32
func.func @cmpi_eq_i16x32(%a: vector<32xi16>, %b: vector<32xi16>) -> vector<32xi1> {
  // CHECK: arith.cmpi eq
  // CHECK-NOT: aievec.aie4.cmp
  %0 = arith.cmpi eq, %a, %b : vector<32xi16>
  return %0 : vector<32xi1>
}

// -----

// -- SELECT --

// CHECK-LABEL: @select_i32x32
// CHECK-SAME: %[[COND:.*]]: vector<32xi1>, %[[T:.*]]: vector<32xi32>, %[[F:.*]]: vector<32xi32>
func.func @select_i32x32(%cond: vector<32xi1>, %t: vector<32xi32>, %f: vector<32xi32>) -> vector<32xi32> {
  // CHECK: %[[V:.*]] = vector.bitcast %[[COND]] : vector<32xi1> to vector<1xi32>
  // CHECK: %[[S:.*]] = vector.extract %[[V]][0] : i32 from vector<1xi32>
  // CHECK: %[[SEL:.*]] = builtin.unrealized_conversion_cast %[[S]] : i32 to ui32
  // Native select takes the false value first and the true value second.
  // CHECK: aievec.aie4.sel %[[F]], %[[T]], %[[SEL]] : vector<32xi32>, ui32, vector<32xi32>
  %0 = arith.select %cond, %t, %f : vector<32xi1>, vector<32xi32>
  return %0 : vector<32xi32>
}

// Unsupported shapes retain generic select.
// CHECK-LABEL: @select_out_of_scope
func.func @select_out_of_scope(%cond: vector<8xi1>, %t: vector<8xi32>, %f: vector<8xi32>) -> vector<8xi32> {
  // CHECK: arith.select
  // CHECK-NOT: aievec.aie4.sel
  %0 = arith.select %cond, %t, %f : vector<8xi1>, vector<8xi32>
  return %0 : vector<8xi32>
}

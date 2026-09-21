// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" | FileCheck %s

// Unsupported element types and vector shapes retain generic operations.

// CHECK-LABEL: @float16
// CHECK-NOT: aievec.aie4
// CHECK: arith.select
// CHECK-NOT: aievec.aie4
// CHECK: return
func.func @float16(%m: vector<16xi1>, %a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xf32> {
  %r = arith.select %m, %a, %b : vector<16xi1>, vector<16xf32>
  return %r : vector<16xf32>
}

// CHECK-LABEL: @float32
// CHECK-NOT: aievec.aie4
// CHECK: arith.select
// CHECK-NOT: aievec.aie4
// CHECK: return
func.func @float32(%m: vector<32xi1>, %a: vector<32xf32>, %b: vector<32xf32>) -> vector<32xf32> {
  %r = arith.select %m, %a, %b : vector<32xi1>, vector<32xf32>
  return %r : vector<32xf32>
}

// CHECK-LABEL: @half32
// CHECK-NOT: aievec.aie4
// CHECK: arith.select
// CHECK-NOT: aievec.aie4
// CHECK: return
func.func @half32(%m: vector<32xi1>, %a: vector<32xf16>, %b: vector<32xf16>) -> vector<32xf16> {
  %r = arith.select %m, %a, %b : vector<32xi1>, vector<32xf16>
  return %r : vector<32xf16>
}

// CHECK-LABEL: @half64
// CHECK-NOT: aievec.aie4
// CHECK: arith.select
// CHECK-NOT: aievec.aie4
// CHECK: return
func.func @half64(%m: vector<64xi1>, %a: vector<64xf16>, %b: vector<64xf16>) -> vector<64xf16> {
  %r = arith.select %m, %a, %b : vector<64xi1>, vector<64xf16>
  return %r : vector<64xf16>
}

// CHECK-LABEL: @scalable
// CHECK-NOT: aievec.aie4
// CHECK: arith.select
// CHECK-NOT: aievec.aie4
// CHECK: return
func.func @scalable(%m: vector<[16]xi1>, %a: vector<[16]xi32>, %b: vector<[16]xi32>) -> vector<[16]xi32> {
  %r = arith.select %m, %a, %b : vector<[16]xi1>, vector<[16]xi32>
  return %r : vector<[16]xi32>
}

// CHECK-LABEL: @rank_two
// CHECK-NOT: aievec.aie4
// CHECK: arith.select
// CHECK-NOT: aievec.aie4
// CHECK: return
func.func @rank_two(%m: vector<4x4xi1>, %a: vector<4x4xi32>, %b: vector<4x4xi32>) -> vector<4x4xi32> {
  %r = arith.select %m, %a, %b : vector<4x4xi1>, vector<4x4xi32>
  return %r : vector<4x4xi32>
}

// CHECK-LABEL: @compare_scalable
// CHECK-NOT: aievec.aie4
// CHECK: arith.cmpi slt
// CHECK-NOT: aievec.aie4
// CHECK: return
func.func @compare_scalable(%a: vector<[16]xi32>, %b: vector<[16]xi32>) -> vector<[16]xi1> {
  %r = arith.cmpi slt, %a, %b : vector<[16]xi32>
  return %r : vector<[16]xi1>
}

// CHECK-LABEL: @min_scalable
// CHECK-NOT: aievec.aie4
// CHECK: arith.minsi
// CHECK-NOT: aievec.aie4
// CHECK: return
func.func @min_scalable(%a: vector<[16]xi32>, %b: vector<[16]xi32>) -> vector<[16]xi32> {
  %r = arith.minsi %a, %b : vector<[16]xi32>
  return %r : vector<[16]xi32>
}

// CHECK-LABEL: @compare_rank_two
// CHECK-NOT: aievec.aie4
// CHECK: arith.cmpi slt
// CHECK-NOT: aievec.aie4
// CHECK: return
func.func @compare_rank_two(%a: vector<4x4xi32>, %b: vector<4x4xi32>) -> vector<4x4xi1> {
  %r = arith.cmpi slt, %a, %b : vector<4x4xi32>
  return %r : vector<4x4xi1>
}

// CHECK-LABEL: @min_rank_two
// CHECK-NOT: aievec.aie4
// CHECK: arith.minsi
// CHECK-NOT: aievec.aie4
// CHECK: return
func.func @min_rank_two(%a: vector<4x4xi32>, %b: vector<4x4xi32>) -> vector<4x4xi32> {
  %r = arith.minsi %a, %b : vector<4x4xi32>
  return %r : vector<4x4xi32>
}

// CHECK-LABEL: @signed
// CHECK-NOT: aievec.aie4
// CHECK: arith.select
// CHECK-NOT: aievec.aie4
// CHECK: return
func.func @signed(%m: vector<16xi1>, %a: vector<16xsi32>, %b: vector<16xsi32>) -> vector<16xsi32> {
  %r = arith.select %m, %a, %b : vector<16xi1>, vector<16xsi32>
  return %r : vector<16xsi32>
}

// CHECK-LABEL: @unsigned
// CHECK-NOT: aievec.aie4
// CHECK: arith.select
// CHECK-NOT: aievec.aie4
// CHECK: return
func.func @unsigned(%m: vector<16xi1>, %a: vector<16xui32>, %b: vector<16xui32>) -> vector<16xui32> {
  %r = arith.select %m, %a, %b : vector<16xi1>, vector<16xui32>
  return %r : vector<16xui32>
}


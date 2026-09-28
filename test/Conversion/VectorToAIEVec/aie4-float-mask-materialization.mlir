// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" --convert-aievec-to-llvm="aie-target=aie4b" --convert-vector-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | aie-translate --mlir-to-llvmir | FileCheck %s

// Float masks use the same packing as integer masks, including the unused
// high bits of the 16-lane carrier and composed signless compare results.

// CHECK-LABEL: define <16 x i1> @float_compare_16(
// CHECK: @llvm.aie4.vadd32.512
// CHECK: trunc i32 {{.*}} to i16
// CHECK: bitcast <1 x i16> {{.*}} to <16 x i1>
// CHECK: ret <16 x i1>
func.func @float_compare_16(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xi1> {
  %m = arith.cmpf olt, %a, %b : vector<16xf32>
  return %m : vector<16xi1>
}

// CHECK-LABEL: define <32 x i1> @float_compare_32(
// CHECK: @llvm.aie4.vadd32.1024
// CHECK: and i32
// CHECK: bitcast <1 x i32> {{.*}} to <32 x i1>
// CHECK: ret <32 x i1>
func.func @float_compare_32(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<32xi1> {
  %m = arith.cmpf oeq, %a, %b : vector<32xf32>
  return %m : vector<32xi1>
}

// CHECK-LABEL: define <64 x i1> @float_compare_64(
// CHECK: @llvm.aie4.vadd16.1024
// CHECK: and i64
// CHECK: xor i64
// CHECK: bitcast <1 x i64> {{.*}} to <64 x i1>
// CHECK: ret <64 x i1>
func.func @float_compare_64(%a: vector<64xbf16>, %b: vector<64xbf16>) -> vector<64xi1> {
  %m = arith.cmpf uno, %a, %b : vector<64xbf16>
  return %m : vector<64xi1>
}

// CHECK-LABEL: define <16 x float> @float_select_16(
// CHECK: bitcast <16 x i1> {{.*}} to <1 x i16>
// CHECK: zext i16 {{.*}} to i32
// CHECK: @llvm.aie4.vadd32.mask.512
// CHECK: ret <16 x float>
func.func @float_select_16(%m: vector<16xi1>, %t: vector<16xf32>, %f: vector<16xf32>) -> vector<16xf32> {
  %r = arith.select %m, %t, %f : vector<16xi1>, vector<16xf32>
  return %r : vector<16xf32>
}

// CHECK-LABEL: define <64 x bfloat> @float_compare_select_64(
// CHECK: @llvm.aie4.vadd16.1024
// CHECK: @llvm.aie4.vadd16.mask.1024
// CHECK: ret <64 x bfloat>
func.func @float_compare_select_64(%a: vector<64xbf16>, %b: vector<64xbf16>) -> vector<64xbf16> {
  %m = arith.cmpf olt, %a, %b : vector<64xbf16>
  %r = arith.select %m, %a, %b : vector<64xi1>, vector<64xbf16>
  return %r : vector<64xbf16>
}

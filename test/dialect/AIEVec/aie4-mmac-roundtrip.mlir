// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s | FileCheck %s
// RUN: aie-opt %s | aie-opt | FileCheck %s
// RUN: aie-opt %s -mlir-print-op-generic | aie-opt | FileCheck %s

// Semantic matmul over one native B tile, for the four dense formats.

// CHECK-LABEL: func.func @matmul_bf16
// CHECK: aievec.aie4.matmul %{{.*}}, %{{.*}}, %{{.*}} : vector<4x32xbf16>, vector<32x64xbf16> into vector<4x64xf32>
func.func @matmul_bf16(%a: vector<4x32xbf16>, %b: vector<32x64xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  %0 = aievec.aie4.matmul %a, %b, %c : vector<4x32xbf16>, vector<32x64xbf16> into vector<4x64xf32>
  return %0 : vector<4x64xf32>
}

// CHECK-LABEL: func.func @matmul_i8
// CHECK: aievec.aie4.matmul %{{.*}}, %{{.*}}, %{{.*}} {lhs_signed = true, rhs_signed = false} : vector<1x64xi8>, vector<64x64xi8> into vector<1x64xi32>
func.func @matmul_i8(%a: vector<1x64xi8>, %b: vector<64x64xi8>, %c: vector<1x64xi32>) -> vector<1x64xi32> {
  %0 = aievec.aie4.matmul %a, %b, %c {lhs_signed = true, rhs_signed = false} : vector<1x64xi8>, vector<64x64xi8> into vector<1x64xi32>
  return %0 : vector<1x64xi32>
}

// CHECK-LABEL: func.func @matmul_i4
// CHECK: aievec.aie4.matmul %{{.*}}, %{{.*}}, %{{.*}} {lhs_signed = true, rhs_signed = true} : vector<16x128xi4>, vector<128x64xi4> into vector<16x64xi32>
func.func @matmul_i4(%a: vector<16x128xi4>, %b: vector<128x64xi4>, %c: vector<16x64xi32>) -> vector<16x64xi32> {
  %0 = aievec.aie4.matmul %a, %b, %c {lhs_signed = true, rhs_signed = true} : vector<16x128xi4>, vector<128x64xi4> into vector<16x64xi32>
  return %0 : vector<16x64xi32>
}

// CHECK-LABEL: func.func @matmul_fp8
// CHECK: aievec.aie4.matmul %{{.*}}, %{{.*}}, %{{.*}} : vector<64x64xf8E4M3FN>, vector<64x64xf8E4M3FN> into vector<64x64xf32>
func.func @matmul_fp8(%a: vector<64x64xf8E4M3FN>, %b: vector<64x64xf8E4M3FN>, %c: vector<64x64xf32>) -> vector<64x64xf32> {
  %0 = aievec.aie4.matmul %a, %b, %c : vector<64x64xf8E4M3FN>, vector<64x64xf8E4M3FN> into vector<64x64xf32>
  return %0 : vector<64x64xf32>
}

// An output-row loop inside the region reuses the resident matrix.
// CHECK-LABEL: func.func @region_bf16_rows
// CHECK: %[[OUT:.*]] = aievec.aie4.matrix_region bf16 %{{.*}} : vector<32x64xbf16>, %{{.*}} : vector<4x64xf32> -> vector<4x64xf32> {
// CHECK: ^bb0(%[[ACC:.*]]: vector<4x64xf32>):
// CHECK: scf.for {{.*}} iter_args
// CHECK: aievec.aie4.mmac bf16 %{{.*}}, %{{.*}} : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
// CHECK: aievec.aie4.matrix_yield %{{.*}} : vector<4x64xf32>
// CHECK: return %[[OUT]]
func.func @region_bf16_rows(%a: vector<4x32xbf16>, %b: vector<32x64xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %out = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16>, %c : vector<4x64xf32> -> vector<4x64xf32> {
  ^bb0(%acc0: vector<4x64xf32>):
    %rows = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %acc0) -> (vector<4x64xf32>) {
      %a_i = vector.extract %a[%i] : vector<32xbf16> from vector<4x32xbf16>
      %c_i = vector.extract %acc[%i] : vector<64xf32> from vector<4x64xf32>
      %r_i = aievec.aie4.mmac bf16 %a_i, %c_i : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
      %next = vector.insert %r_i, %acc [%i] : vector<64xf32> into vector<4x64xf32>
      scf.yield %next : vector<4x64xf32>
    }
    aievec.aie4.matrix_yield %rows : vector<4x64xf32>
  }
  return %out : vector<4x64xf32>
}

// CHECK-LABEL: func.func @region_int8
// CHECK: aievec.aie4.matrix_region int8 %{{.*}} : vector<64x64xi8>, %{{.*}} : vector<64xi32> -> vector<64xi32> attributes {rhs_signed = true} {
// CHECK: aievec.aie4.mmac int8 %{{.*}}, %{{.*}} {lhs_signed = false, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
func.func @region_int8(%a: vector<64xi8>, %b: vector<64x64xi8>, %c: vector<64xi32>) -> vector<64xi32> {
  %out = aievec.aie4.matrix_region int8 %b : vector<64x64xi8>, %c : vector<64xi32> -> vector<64xi32> attributes {rhs_signed = true} {
  ^bb0(%acc: vector<64xi32>):
    %r = aievec.aie4.mmac int8 %a, %acc {lhs_signed = false, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
    aievec.aie4.matrix_yield %r : vector<64xi32>
  }
  return %out : vector<64xi32>
}

// The accumulator is optional: without it the hardware zeroes it.
// CHECK-LABEL: func.func @region_int4_zero_acc
// CHECK: aievec.aie4.matrix_region int4 %{{.*}} : vector<128x64xi4> -> vector<64xi32> attributes {rhs_signed = true} {
// CHECK: aievec.aie4.mmac int4 %{{.*}} {lhs_signed = true, rhs_signed = true} : vector<128xi4> -> vector<64xi32>
func.func @region_int4_zero_acc(%a: vector<128xi4>, %b: vector<128x64xi4>) -> vector<64xi32> {
  %out = aievec.aie4.matrix_region int4 %b : vector<128x64xi4> -> vector<64xi32> attributes {rhs_signed = true} {
    %r = aievec.aie4.mmac int4 %a {lhs_signed = true, rhs_signed = true} : vector<128xi4> -> vector<64xi32>
    aievec.aie4.matrix_yield %r : vector<64xi32>
  }
  return %out : vector<64xi32>
}

// Two rows, two accumulators, no results beyond the yielded pair.
// CHECK-LABEL: func.func @region_fp8_two_rows
// CHECK: aievec.aie4.matrix_region fp8 %{{.*}} : vector<64x64xf8E4M3FN>, %{{.*}}, %{{.*}} : vector<64xf32>, vector<64xf32> -> vector<64xf32>, vector<64xf32> {
// CHECK-COUNT-2: aievec.aie4.mmac fp8
func.func @region_fp8_two_rows(%a0: vector<64xf8E4M3FN>, %a1: vector<64xf8E4M3FN>, %b: vector<64x64xf8E4M3FN>, %c0: vector<64xf32>, %c1: vector<64xf32>) -> (vector<64xf32>, vector<64xf32>) {
  %r:2 = aievec.aie4.matrix_region fp8 %b : vector<64x64xf8E4M3FN>, %c0, %c1 : vector<64xf32>, vector<64xf32> -> vector<64xf32>, vector<64xf32> {
  ^bb0(%x0: vector<64xf32>, %x1: vector<64xf32>):
    %y0 = aievec.aie4.mmac fp8 %a0, %x0 : vector<64xf8E4M3FN>, vector<64xf32> -> vector<64xf32>
    %y1 = aievec.aie4.mmac fp8 %a1, %x1 : vector<64xf8E4M3FN>, vector<64xf32> -> vector<64xf32>
    aievec.aie4.matrix_yield %y0, %y1 : vector<64xf32>, vector<64xf32>
  }
  return %r#0, %r#1 : vector<64xf32>, vector<64xf32>
}

// A region with no results.
// CHECK-LABEL: func.func @region_no_results
// CHECK: aievec.aie4.matrix_region bf16 %{{.*}} : vector<32x64xbf16> {
// CHECK-NEXT: aievec.aie4.matrix_yield
// CHECK-NEXT: }
func.func @region_no_results(%b: vector<32x64xbf16>) {
  aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> {
    aievec.aie4.matrix_yield
  }
  return
}

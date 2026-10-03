// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -aievec-aie4-expand-matmul | FileCheck %s

// aievec.aie4.matmul becomes one matrix region holding the B tile, with one
// row MMAC per output row. Rows are unrolled with static positions.

// CHECK-LABEL: func.func @bf16_rows
// CHECK-SAME: (%[[A:.*]]: vector<3x32xbf16>, %[[B:.*]]: vector<32x64xbf16>, %[[C:.*]]: vector<3x64xf32>)
// CHECK: %[[R:.*]] = aievec.aie4.matrix_region bf16 %[[B]] : vector<32x64xbf16>, %[[C]] : vector<3x64xf32> -> vector<3x64xf32> {
// CHECK-NEXT: ^bb0(%[[ACC:.*]]: vector<3x64xf32>):
// CHECK-NEXT: %[[A0:.*]] = vector.extract %[[A]][0] : vector<32xbf16> from vector<3x32xbf16>
// CHECK-NEXT: %[[C0:.*]] = vector.extract %[[ACC]][0] : vector<64xf32> from vector<3x64xf32>
// CHECK-NEXT: %[[M0:.*]] = aievec.aie4.mmac bf16 %[[A0]], %[[C0]] : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
// CHECK-NEXT: %[[I0:.*]] = vector.insert %[[M0]], %[[ACC]] [0] : vector<64xf32> into vector<3x64xf32>
// CHECK-NEXT: %[[A1:.*]] = vector.extract %[[A]][1]
// CHECK-NEXT: %[[C1:.*]] = vector.extract %[[ACC]][1]
// CHECK-NEXT: %[[M1:.*]] = aievec.aie4.mmac bf16 %[[A1]], %[[C1]]
// CHECK-NEXT: %[[I1:.*]] = vector.insert %[[M1]], %[[I0]] [1]
// CHECK-NEXT: %[[A2:.*]] = vector.extract %[[A]][2]
// CHECK-NEXT: %[[C2:.*]] = vector.extract %[[ACC]][2]
// CHECK-NEXT: %[[M2:.*]] = aievec.aie4.mmac bf16 %[[A2]], %[[C2]]
// CHECK-NEXT: %[[I2:.*]] = vector.insert %[[M2]], %[[I1]] [2]
// CHECK-NEXT: aievec.aie4.matrix_yield %[[I2]] : vector<3x64xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[R]]
// CHECK-NOT: aievec.aie4.matmul
func.func @bf16_rows(%a: vector<3x32xbf16>, %b: vector<32x64xbf16>, %c: vector<3x64xf32>) -> vector<3x64xf32> {
  %0 = aievec.aie4.matmul %a, %b, %c : vector<3x32xbf16>, vector<32x64xbf16> into vector<3x64xf32>
  return %0 : vector<3x64xf32>
}

// -----

// Signedness moves to the region (B) and to each MMAC (A and B).
// CHECK-LABEL: func.func @int4_signs
// CHECK: aievec.aie4.matrix_region int4 %{{.*}} : vector<128x64xi4>, %{{.*}} : vector<1x64xi32> -> vector<1x64xi32> attributes {rhs_signed = false} {
// CHECK: aievec.aie4.mmac int4 %{{.*}}, %{{.*}} {lhs_signed = true, rhs_signed = false} : vector<128xi4>, vector<64xi32> -> vector<64xi32>
// CHECK-NOT: aievec.aie4.mmac
func.func @int4_signs(%a: vector<1x128xi4>, %b: vector<128x64xi4>, %c: vector<1x64xi32>) -> vector<1x64xi32> {
  %0 = aievec.aie4.matmul %a, %b, %c {lhs_signed = true, rhs_signed = false} : vector<1x128xi4>, vector<128x64xi4> into vector<1x64xi32>
  return %0 : vector<1x64xi32>
}

// -----

// K tiles are chained by the producer: one region per B tile, the partial
// accumulator flowing from one to the next.
// CHECK-LABEL: func.func @two_k_tiles
// CHECK: %[[R0:.*]] = aievec.aie4.matrix_region fp8 %{{.*}}, %{{.*}} : vector<2x64xf32>
// CHECK: %[[R1:.*]] = aievec.aie4.matrix_region fp8 %{{.*}}, %[[R0]] : vector<2x64xf32>
// CHECK: return %[[R1]]
func.func @two_k_tiles(%a0: vector<2x64xf8E4M3FN>, %a1: vector<2x64xf8E4M3FN>, %b0: vector<64x64xf8E4M3FN>, %b1: vector<64x64xf8E4M3FN>, %c: vector<2x64xf32>) -> vector<2x64xf32> {
  %0 = aievec.aie4.matmul %a0, %b0, %c : vector<2x64xf8E4M3FN>, vector<64x64xf8E4M3FN> into vector<2x64xf32>
  %1 = aievec.aie4.matmul %a1, %b1, %0 : vector<2x64xf8E4M3FN>, vector<64x64xf8E4M3FN> into vector<2x64xf32>
  return %1 : vector<2x64xf32>
}


// -----

// 64 rows is the largest M the lowering accepts.
// CHECK-LABEL: func.func @max_rows
// CHECK-COUNT-64: aievec.aie4.mmac int8
// CHECK-NOT: aievec.aie4.mmac
func.func @max_rows(%a: vector<64x64xi8>, %b: vector<64x64xi8>, %c: vector<64x64xi32>) -> vector<64x64xi32> {
  %0 = aievec.aie4.matmul %a, %b, %c {lhs_signed = true, rhs_signed = true} : vector<64x64xi8>, vector<64x64xi8> into vector<64x64xi32>
  return %0 : vector<64x64xi32>
}

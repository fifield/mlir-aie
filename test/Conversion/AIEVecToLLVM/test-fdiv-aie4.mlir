//===- test-fdiv-aie4.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm="aie-target=aie4b" | FileCheck %s

// Scalar stays put; only the vector form is claimed.

// CHECK-LABEL: @scalar_f32_fdiv
// CHECK-SAME: %[[LHS:.*]]: f32, %[[RHS:.*]]: f32
func.func @scalar_f32_fdiv(%lhs : f32, %rhs : f32) -> f32 {
  // CHECK: %[[RES:.*]] = arith.divf %[[LHS]], %[[RHS]] : f32
  // CHECK: return %[[RES]] : f32
  %0 = arith.divf %lhs, %rhs : f32
  return %0 : f32
}

// -----

// An ordinary divide keeps a real division. INV flushes a subnormal
// reciprocal to zero, so a * INV(b) is not a * (1/b) when b is a large
// normal: 2^127 / 2^127 through INV is 0, not 1.

// CHECK: llvm.func @__aie4_scalar_fdiv
// CHECK-SAME: attributes {passthrough = ["noinline"]}
// CHECK: %[[DIV:.*]] = arith.divf %arg0, %arg1 : f32
// CHECK: llvm.return %[[DIV]] : f32
// CHECK-NOT: xllvm.intr.aie4.inv
// CHECK-LABEL: @vector_4xf32_fdiv
// CHECK-SAME: %[[LHS:.*]]: vector<4xf32>, %[[RHS:.*]]: vector<4xf32>
func.func @vector_4xf32_fdiv(%lhs : vector<4xf32>, %rhs : vector<4xf32>) -> vector<4xf32> {
  // CHECK: %[[POISON:.*]] = llvm.mlir.poison : vector<4xf32>
  // CHECK-COUNT-4: llvm.call @__aie4_scalar_fdiv
  %0 = arith.divf %lhs, %rhs : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// One lane is still a vector, and the legalization rule does not exempt it.

// CHECK: llvm.func @__aie4_scalar_fdiv
// CHECK-LABEL: @vector_1xf32_fdiv
func.func @vector_1xf32_fdiv(%lhs : vector<1xf32>, %rhs : vector<1xf32>) -> vector<1xf32> {
  // CHECK-COUNT-1: llvm.call @__aie4_scalar_fdiv
  %0 = arith.divf %lhs, %rhs : vector<1xf32>
  return %0 : vector<1xf32>
}

// -----

// CHECK: llvm.func @__aie4_scalar_fdiv
// CHECK-LABEL: @vector_32xf32_fdiv
func.func @vector_32xf32_fdiv(%lhs : vector<32xf32>, %rhs : vector<32xf32>) -> vector<32xf32> {
  // CHECK-COUNT-32: llvm.call @__aie4_scalar_fdiv
  // CHECK-NOT: llvm.call @__aie4_scalar_fdiv
  %0 = arith.divf %lhs, %rhs : vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

// The legalization rule names f32, so bf16 is neither claimed nor rejected.

// CHECK-LABEL: @vector_32xbf16_fdiv_untouched
// CHECK: arith.divf
// CHECK-NOT: __aie4_scalar_fdiv
func.func @vector_32xbf16_fdiv_untouched(%lhs : vector<32xbf16>, %rhs : vector<32xbf16>) -> vector<32xbf16> {
  %0 = arith.divf %lhs, %rhs : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

// One helper per module, whatever the width and however many divides reach
// it -- a second definition would not link.

// CHECK-COUNT-1: llvm.func @__aie4_scalar_fdiv
// CHECK-NOT: llvm.func @__aie4_scalar_fdiv
func.func @two_widths_one_helper(%a : vector<4xf32>, %b : vector<8xf32>)
    -> (vector<4xf32>, vector<8xf32>) {
  %0 = arith.divf %a, %a : vector<4xf32>
  %1 = arith.divf %b, %b : vector<8xf32>
  return %0, %1 : vector<4xf32>, vector<8xf32>
}

func.func @second_func_reuses_helper(%a : vector<4xf32>) -> vector<4xf32> {
  %0 = arith.divf %a, %a : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// INV needs both `arcp` (a reciprocal may replace the division) and `afn`
// (an approximation is acceptable). Either flag alone keeps a real division.

// CHECK: llvm.func @__aie4_scalar_fdiv_inv
// CHECK: %[[INV:.*]] = "xllvm.intr.aie4.inv"(%arg1) : (f32) -> f32
// CHECK: %[[MUL:.*]] = arith.mulf %arg0, %[[INV]] : f32
// CHECK: llvm.return %[[MUL]] : f32
// CHECK-LABEL: @vector_4xf32_fdiv_arcp_afn
// CHECK-COUNT-4: llvm.call @__aie4_scalar_fdiv_inv
func.func @vector_4xf32_fdiv_arcp_afn(%lhs : vector<4xf32>, %rhs : vector<4xf32>) -> vector<4xf32> {
  %0 = arith.divf %lhs, %rhs fastmath<arcp,afn> : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// fast includes both.
// CHECK-LABEL: @vector_4xf32_fdiv_fast
// CHECK-COUNT-4: llvm.call @__aie4_scalar_fdiv_inv
func.func @vector_4xf32_fdiv_fast(%lhs : vector<4xf32>, %rhs : vector<4xf32>) -> vector<4xf32> {
  %0 = arith.divf %lhs, %rhs fastmath<fast> : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-NOT: __aie4_scalar_fdiv_inv
// CHECK-LABEL: @vector_4xf32_fdiv_arcp_only
// CHECK-COUNT-4: llvm.call @__aie4_scalar_fdiv(
func.func @vector_4xf32_fdiv_arcp_only(%lhs : vector<4xf32>, %rhs : vector<4xf32>) -> vector<4xf32> {
  %0 = arith.divf %lhs, %rhs fastmath<arcp> : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-NOT: __aie4_scalar_fdiv_inv
// CHECK-LABEL: @vector_4xf32_fdiv_afn_only
// CHECK-COUNT-4: llvm.call @__aie4_scalar_fdiv(
func.func @vector_4xf32_fdiv_afn_only(%lhs : vector<4xf32>, %rhs : vector<4xf32>) -> vector<4xf32> {
  %0 = arith.divf %lhs, %rhs fastmath<afn> : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// Both forms in one module get distinct symbols. getOrCreateScalarHelperFunc
// caches by name, so sharing one would make the result depend on which
// division was converted first.

// CHECK-DAG: llvm.func @__aie4_scalar_fdiv(
// CHECK-DAG: llvm.func @__aie4_scalar_fdiv_inv(
func.func @both_forms(%a : vector<4xf32>, %b : vector<4xf32>)
    -> (vector<4xf32>, vector<4xf32>) {
  %0 = arith.divf %a, %b : vector<4xf32>
  %1 = arith.divf %a, %b fastmath<arcp,afn> : vector<4xf32>
  return %0, %1 : vector<4xf32>, vector<4xf32>
}

//
// target-parsing-aie4b-lower-mulf.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The positive pin for AIE4's Vector->AIEVec lowering: `arith.mulf` on
// vector<32xf32> -- one full 1024-bit AIE4 vector register -- becomes
// `aievec.aie4.mul_ewise_f32`, AIE4's own op.
//
// The pattern is deliberately narrow. Only this exact shape is claimed;
// every other arith.mulf stays legal and falls through to the generic LLVM
// path, which is what the negative cases below check.
//
// RUN: aie-opt %s --lower-vector-to-aievec="aie-target=aie4b" | FileCheck %s
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" | FileCheck %s

// CHECK-LABEL: func.func @mulf_v32f32
// CHECK: aievec.aie4.mul_ewise_f32 %arg0, %arg1 : vector<32xf32>
// CHECK-NOT: arith.mulf
func.func @mulf_v32f32(%a : vector<32xf32>, %b : vector<32xf32>) -> vector<32xf32> {
  %0 = arith.mulf %a, %b : vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

// Wrong lane count: 16 lanes is half a register, and no AIE4 intrinsic
// claims it. Must be left alone.
// CHECK-LABEL: func.func @mulf_v16f32
// CHECK: arith.mulf
// CHECK-NOT: aievec.aie4.mul_ewise_f32
func.func @mulf_v16f32(%a : vector<16xf32>, %b : vector<16xf32>) -> vector<16xf32> {
  %0 = arith.mulf %a, %b : vector<16xf32>
  return %0 : vector<16xf32>
}

// -----

// Right lane count, wrong element type.
// CHECK-LABEL: func.func @mulf_v32bf16
// CHECK: arith.mulf
// CHECK-NOT: aievec.aie4.mul_ewise_f32
func.func @mulf_v32bf16(%a : vector<32xbf16>, %b : vector<32xbf16>) -> vector<32xbf16> {
  %0 = arith.mulf %a, %b : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

// Scalar f32: not a vector at all.
// CHECK-LABEL: func.func @mulf_scalar
// CHECK: arith.mulf
// CHECK-NOT: aievec.aie4.mul_ewise_f32
func.func @mulf_scalar(%a : f32, %b : f32) -> f32 {
  %0 = arith.mulf %a, %b : f32
  return %0 : f32
}

// -----

// A scalable vector has a minimum of 32 lanes, not a fixed 1024-bit size.
// CHECK-LABEL: func.func @mulf_scalable
// CHECK: arith.mulf %arg0, %arg1 : vector<[32]xf32>
// CHECK-NOT: aievec.aie4.mul_ewise_f32
func.func @mulf_scalable(%a : vector<[32]xf32>, %b : vector<[32]xf32>) -> vector<[32]xf32> {
  %0 = arith.mulf %a, %b : vector<[32]xf32>
  return %0 : vector<[32]xf32>
}

// -----

// Having 32 elements is insufficient: the intrinsic requires rank one.
// CHECK-LABEL: func.func @mulf_rank_two
// CHECK: arith.mulf %arg0, %arg1 : vector<4x8xf32>
// CHECK-NOT: aievec.aie4.mul_ewise_f32
func.func @mulf_rank_two(%a : vector<4x8xf32>, %b : vector<4x8xf32>) -> vector<4x8xf32> {
  %0 = arith.mulf %a, %b : vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Native reductions and their XLLVM wrappers require fixed rank-one vectors.

func.func @aievec_scalable(%a: vector<[32]xi32>) -> i32 {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "aievec.aie4.reduce"(%a) {kind = #aievec.aie4_reduce_kind<add>} : (vector<[32]xi32>) -> i32
  return %0 : i32
}

// -----

func.func @aievec_rank_two(%a: vector<4x8xf32>) -> f32 {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "aievec.aie4.reduce"(%a) {kind = #aievec.aie4_reduce_kind<add>} : (vector<4x8xf32>) -> f32
  return %0 : f32
}

// -----

func.func @xllvm_int_scalable(%a: vector<[32]xi32>, %seed: i32) -> i32 {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.add.reduce.I1024"(%a, %seed) : (vector<[32]xi32>, i32) -> i32
  return %0 : i32
}

// -----

func.func @xllvm_float_rank_two(%a: vector<4x8xf32>) -> f32 {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.add.reduce.FP1024"(%a) : (vector<4x8xf32>) -> f32
  return %0 : f32
}

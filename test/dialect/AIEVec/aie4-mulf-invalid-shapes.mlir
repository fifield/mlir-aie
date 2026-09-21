// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics

func.func @scalable(%a: vector<[32]xf32>, %b: vector<[32]xf32>) -> vector<[32]xf32> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = aievec.aie4.mul_ewise_f32 %a, %b : vector<[32]xf32>
  return %0 : vector<[32]xf32>
}

// -----

func.func @rank_two(%a: vector<4x8xf32>, %b: vector<4x8xf32>) -> vector<4x8xf32> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = aievec.aie4.mul_ewise_f32 %a, %b : vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// -----

// Generic syntax permits testing the result constraint independently.
func.func @scalable_result(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<[32]xf32> {
  // expected-error @+1 {{result #0 must be fixed-length vector}}
  %0 = "aievec.aie4.mul_ewise_f32"(%a, %b) : (vector<32xf32>, vector<32xf32>) -> vector<[32]xf32>
  return %0 : vector<[32]xf32>
}

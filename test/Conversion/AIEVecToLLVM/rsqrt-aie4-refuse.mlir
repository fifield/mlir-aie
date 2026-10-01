// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" -verify-diagnostics

// INVSQRT is approximate and reads a subnormal input as zero, so strict f32
// rsqrt has no AIE4 lowering and is diagnosed here.

func.func @rsqrt_strict_scalar(%a: f32) -> f32 {
  // expected-warning @+2 {{AIE4: math.rsqrt requires fastmath<afn>; AIE4 has only the approximate INVSQRT lowering and no strict lowering}}
  // expected-error @+1 {{failed to legalize operation 'math.rsqrt'}}
  %0 = math.rsqrt %a : f32
  return %0 : f32
}

// -----

func.func @rsqrt_strict_vector(%a: vector<32xf32>) -> vector<32xf32> {
  // expected-warning @+2 {{requires fastmath<afn>}}
  // expected-error @+1 {{failed to legalize operation 'math.rsqrt'}}
  %0 = math.rsqrt %a : vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

// Other flags do not stand in for afn.
func.func @rsqrt_arcp_only(%a: vector<4xf32>) -> vector<4xf32> {
  // expected-warning @+2 {{requires fastmath<afn>}}
  // expected-error @+1 {{failed to legalize operation 'math.rsqrt'}}
  %0 = math.rsqrt %a fastmath<arcp,contract> : vector<4xf32>
  return %0 : vector<4xf32>
}

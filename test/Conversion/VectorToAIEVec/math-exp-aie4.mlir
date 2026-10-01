// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie4b" --convert-vector-to-aievec="aie-target=aie4b" -split-input-file -verify-diagnostics | FileCheck %s

// afn exp is VEXP2 on x * log2(e), with the product in f32.

// CHECK-LABEL: @exp_f32
// CHECK-SAME: (%[[X:.*]]: vector<32xf32>)
// CHECK-DAG: %[[L:.*]] = arith.constant dense<1.44269502> : vector<32xf32>
// CHECK-DAG: %[[BIAS:.*]] = arith.constant dense<895.000244> : vector<32xf32>
// CHECK: %[[T:.*]] = aievec.aie4.mul_ewise_f32 %[[X]], %[[L]]
// CHECK: %[[B:.*]] = arith.addf %[[T]], %[[BIAS]] : vector<32xf32>
// CHECK: %[[R:.*]] = aievec.aie4.biased_exp2 %[[B]] : vector<32xf32>, vector<32xf32>
// CHECK: return %[[R]]
func.func @exp_f32(%x: vector<32xf32>) -> vector<32xf32> {
  %0 = math.exp %x fastmath<afn> : vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

// CHECK-LABEL: @exp_bf16
// CHECK-SAME: (%[[X:.*]]: vector<32xbf16>)
// CHECK-DAG: %[[L:.*]] = arith.constant dense<1.44269502> : vector<32xf32>
// CHECK-DAG: %[[BIAS:.*]] = arith.constant dense<895.000976> : vector<32xf32>
// CHECK-DAG: %[[W:.*]] = aievec.aie4.conv_bf16_f32 %[[X]]
// CHECK: %[[T:.*]] = aievec.aie4.mul_ewise_f32 %[[W]], %[[L]]
// CHECK: %[[B:.*]] = arith.addf %[[T]], %[[BIAS]] : vector<32xf32>
// CHECK: %[[R:.*]] = aievec.aie4.biased_exp2 %[[B]] : vector<32xf32>, vector<32xbf16>
// CHECK: return %[[R]]
func.func @exp_bf16(%x: vector<32xbf16>) -> vector<32xbf16> {
  %0 = math.exp %x fastmath<afn> : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

// Wider vectors split into 32-lane chunks.
// CHECK-LABEL: @exp_bf16_64
// CHECK-COUNT-2: aievec.aie4.biased_exp2
// CHECK-NOT: math.exp
func.func @exp_bf16_64(%x: vector<64xbf16>) -> vector<64xbf16> {
  %0 = math.exp %x fastmath<fast> : vector<64xbf16>
  return %0 : vector<64xbf16>
}

// -----

// Scalar exp is not claimed.
// CHECK-LABEL: @exp_scalar
// CHECK: math.exp %{{.*}} fastmath<afn> : f32
func.func @exp_scalar(%x: f32) -> f32 {
  %0 = math.exp %x fastmath<afn> : f32
  return %0 : f32
}

// -----

func.func @exp_strict(%x: vector<32xf32>) -> vector<32xf32> {
  // expected-warning @+2 {{AIE4: math.exp requires fastmath<afn>; AIE4 has only the approximate VEXP2 lowering and no strict lowering}}
  // expected-error @+1 {{failed to legalize operation 'math.exp' that was explicitly marked illegal}}
  %0 = math.exp %x : vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @exp_16_lanes(%x: vector<16xbf16>) -> vector<16xbf16> {
  // expected-warning @+2 {{requires a multiple of 32 lanes, not 16}}
  // expected-error @+1 {{failed to legalize operation 'math.exp'}}
  %0 = math.exp %x fastmath<afn> : vector<16xbf16>
  return %0 : vector<16xbf16>
}

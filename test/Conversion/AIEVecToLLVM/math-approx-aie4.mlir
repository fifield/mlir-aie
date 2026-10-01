// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -convert-aievec-to-llvm="aie-target=aie4b" | FileCheck %s

// The biased_exp2 result type selects the VEXP2 form. Neither conversion adds
// or removes the input bias.

// CHECK-LABEL: @biased_exp2_f32
// CHECK-SAME: (%[[X:.*]]: vector<32xf32>)
// CHECK-NEXT: %[[R:.*]] = "xllvm.intr.aie4.exp2.bf20"(%[[X]]) : (vector<32xf32>) -> vector<32xf32>
// CHECK-NEXT: return %[[R]]
func.func @biased_exp2_f32(%x: vector<32xf32>) -> vector<32xf32> {
  %0 = aievec.aie4.biased_exp2 %x : vector<32xf32>, vector<32xf32>
  return %0 : vector<32xf32>
}

// CHECK-LABEL: @biased_exp2_bf16
// CHECK-SAME: (%[[X:.*]]: vector<32xf32>)
// CHECK-NEXT: %[[R:.*]] = "xllvm.intr.aie4.exp2"(%[[X]]) : (vector<32xf32>) -> vector<32xbf16>
// CHECK-NEXT: return %[[R]]
func.func @biased_exp2_bf16(%x: vector<32xf32>) -> vector<32xbf16> {
  %0 = aievec.aie4.biased_exp2 %x : vector<32xf32>, vector<32xbf16>
  return %0 : vector<32xbf16>
}

// CHECK-LABEL: @tanh
// CHECK-SAME: (%[[X:.*]]: vector<32xf32>)
// CHECK-NEXT: %[[R:.*]] = "xllvm.intr.aie4.tanh"(%[[X]]) : (vector<32xf32>) -> vector<32xbf16>
// CHECK-NEXT: return %[[R]]
func.func @tanh(%x: vector<32xf32>) -> vector<32xbf16> {
  %0 = aievec.aie4.tanh %x : vector<32xf32>, vector<32xbf16>
  return %0 : vector<32xbf16>
}

// CHECK-LABEL: @inv
// CHECK-SAME: (%[[X:.*]]: f32)
// CHECK-NEXT: %[[R:.*]] = "xllvm.intr.aie4.inv"(%[[X]]) : (f32) -> f32
// CHECK-NEXT: return %[[R]]
func.func @inv(%x: f32) -> f32 {
  %0 = aievec.aie4.inv %x : f32
  return %0 : f32
}

//===- test-bf16-emulation-aie4.mlir - bf16 emulation is off on AIE4 -*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The bf16-emulation option is accepted on AIE4 and does nothing. AIE2 and
// AIE2P have bf16 vector arithmetic and no f32 vector ALU, so demoting f32 to
// bf16 buys the only vector path there is. On AIE4 it would cost 16 mantissa
// bits and produce an op the part does not implement: there is no bf16 vector
// adder at any width, and arith.addf on bf16 reaches Peano as G_FADD <N x s16>
// and fails to legalize. f32 add, subtract and multiply are all native.

// RUN: aie-opt %s -split-input-file --canonicalize-vector-for-aievec="aie-target=aie4b bf16-emulation" | FileCheck %s

// The two ops with no bf16 vector form on AIE4. Demoting either is what broke
// programming_examples/.../vector_add and vector_sub under BF16_EMULATION=1.

// CHECK-LABEL: func @addf_stays_f32
// CHECK: arith.addf %{{.*}}, %{{.*}} : vector<32xf32>
// CHECK-NOT: arith.truncf
// CHECK-NOT: vector<32xbf16>
func.func @addf_stays_f32(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<32xf32> {
  %0 = arith.addf %a, %b : vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

// CHECK-LABEL: func @subf_stays_f32
// CHECK: arith.subf %{{.*}}, %{{.*}} : vector<32xf32>
// CHECK-NOT: vector<32xbf16>
func.func @subf_stays_f32(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<32xf32> {
  %0 = arith.subf %a, %b : vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

// Multiply does have a bf16 form on AIE4, and is still left alone: the option
// is off for the target as a whole, not per op.

// CHECK-LABEL: func @mulf_stays_f32
// CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<32xf32>
// CHECK-NOT: vector<32xbf16>
func.func @mulf_stays_f32(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<32xf32> {
  %0 = arith.mulf %a, %b : vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

// CHECK-LABEL: func @maxf_stays_f32
// CHECK: arith.maximumf %{{.*}}, %{{.*}} : vector<32xf32>
// CHECK-NOT: vector<32xbf16>
func.func @maxf_stays_f32(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<32xf32> {
  %0 = arith.maximumf %a, %b : vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

// An operand already in bf16 is untouched either way -- the demotion only ever
// rewrote f32 ops, so this pins that nothing else started handling them.

// CHECK-LABEL: func @bf16_input_untouched
// CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<32xbf16>
func.func @bf16_input_untouched(%a: vector<32xbf16>, %b: vector<32xbf16>) -> vector<32xbf16> {
  %0 = arith.mulf %a, %b : vector<32xbf16>
  return %0 : vector<32xbf16>
}

//===- truncf-bf16-aie4.mlir - AIE4 f32 -> bf16 truncf ----------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" | FileCheck %s
// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s --check-prefix=AIE2P

// CHECK-LABEL: @trunc_default
// CHECK: "xllvm.intr.aie4.v32accfloat.to.v32bfloat16"(%{{.*}}) : (vector<32xf32>) -> vector<32xbf16>
// CHECK-NOT: arith.truncf
// AIE2P-LABEL: @trunc_default
// AIE2P: arith.truncf
func.func @trunc_default(%a: vector<32xf32>) -> vector<32xbf16> {
  %0 = arith.truncf %a : vector<32xf32> to vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

// CHECK-LABEL: @trunc_rne
// CHECK: "xllvm.intr.aie4.v32accfloat.to.v32bfloat16"
func.func @trunc_rne(%a: vector<32xf32>) -> vector<32xbf16> {
  %0 = arith.truncf %a to_nearest_even : vector<32xf32> to vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

// The conversion rounds by crRnd, so a truncf that names another mode is left
// alone.
// CHECK-LABEL: @trunc_downward
// CHECK: arith.truncf %{{.*}} downward
// CHECK-NOT: xllvm.intr.aie4.v32accfloat.to.v32bfloat16
func.func @trunc_downward(%a: vector<32xf32>) -> vector<32xbf16> {
  %0 = arith.truncf %a downward : vector<32xf32> to vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

// Shapes without the native conversion are left alone.
// CHECK-LABEL: @trunc_unclaimed
// CHECK: arith.truncf %{{.*}} : vector<16xf32> to vector<16xbf16>
// CHECK: arith.truncf %{{.*}} : f32 to bf16
// CHECK: arith.truncf %{{.*}} : vector<32xf32> to vector<32xf16>
// CHECK-NOT: xllvm.intr.aie4.v32accfloat.to.v32bfloat16
func.func @trunc_unclaimed(%a: vector<16xf32>, %s: f32, %h: vector<32xf32>) -> (vector<16xbf16>, bf16, vector<32xf16>) {
  %0 = arith.truncf %a : vector<16xf32> to vector<16xbf16>
  %1 = arith.truncf %s : f32 to bf16
  %2 = arith.truncf %h : vector<32xf32> to vector<32xf16>
  return %0, %1, %2 : vector<16xbf16>, bf16, vector<32xf16>
}

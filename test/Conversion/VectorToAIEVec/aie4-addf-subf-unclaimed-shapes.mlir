// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" -split-input-file | FileCheck %s

// Shapes no AIE4 pattern claims stay legal and reach generic lowering.
// Marking them illegal with nothing to rewrite them aborts the conversion.

// CHECK-LABEL: @addf_f32_16
// CHECK: arith.addf {{.*}} : vector<16xf32>
func.func @addf_f32_16(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xf32> {
  %r = arith.addf %a, %b : vector<16xf32>
  return %r : vector<16xf32>
}

// -----

// CHECK-LABEL: @addf_bf16_32
// CHECK: arith.addf {{.*}} : vector<32xbf16>
func.func @addf_bf16_32(%a: vector<32xbf16>, %b: vector<32xbf16>) -> vector<32xbf16> {
  %r = arith.addf %a, %b : vector<32xbf16>
  return %r : vector<32xbf16>
}

// -----

// One lane is still a vector, and it is what a design emits when it pins its
// width to stay off the vector datapath.
// CHECK-LABEL: @addf_bf16_1
// CHECK: arith.addf {{.*}} : vector<1xbf16>
func.func @addf_bf16_1(%a: vector<1xbf16>, %b: vector<1xbf16>) -> vector<1xbf16> {
  %r = arith.addf %a, %b : vector<1xbf16>
  return %r : vector<1xbf16>
}

// -----

// CHECK-LABEL: @subf_f32_16
// CHECK: arith.subf {{.*}} : vector<16xf32>
func.func @subf_f32_16(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xf32> {
  %r = arith.subf %a, %b : vector<16xf32>
  return %r : vector<16xf32>
}

// -----

// CHECK-LABEL: @subf_bf16_32
// CHECK: arith.subf {{.*}} : vector<32xbf16>
func.func @subf_bf16_32(%a: vector<32xbf16>, %b: vector<32xbf16>) -> vector<32xbf16> {
  %r = arith.subf %a, %b : vector<32xbf16>
  return %r : vector<32xbf16>
}

// -----

// CHECK-LABEL: @subf_f32_1
// CHECK: arith.subf {{.*}} : vector<1xf32>
func.func @subf_f32_1(%a: vector<1xf32>, %b: vector<1xf32>) -> vector<1xf32> {
  %r = arith.subf %a, %b : vector<1xf32>
  return %r : vector<1xf32>
}

// -----

// Scalars were never claimed and must not become so.
// CHECK-LABEL: @addsubf_scalar
// CHECK: arith.addf {{.*}} : f32
// CHECK: arith.subf {{.*}} : f32
func.func @addsubf_scalar(%a: f32, %b: f32) -> f32 {
  %s = arith.addf %a, %b : f32
  %d = arith.subf %s, %b : f32
  return %d : f32
}

// -----

// The claimed shape is still claimed: a <32xf32> MAC root with contract on
// both operands fuses.
// CHECK-LABEL: @macf_claimed
// CHECK: aievec.aie4.mac_ewise_f32
// CHECK-NOT: arith.addf
func.func @macf_claimed(%a: vector<32xf32>, %b: vector<32xf32>,
                        %c: vector<32xf32>) -> vector<32xf32> {
  %m = arith.mulf %a, %b fastmath<contract> : vector<32xf32>
  %s = arith.addf %c, %m fastmath<contract> : vector<32xf32>
  return %s : vector<32xf32>
}

// -----

// A <32xf32> add that is not a MAC root was legal before and stays legal.
// CHECK-LABEL: @addf_f32_32_plain
// CHECK: arith.addf {{.*}} : vector<32xf32>
func.func @addf_f32_32_plain(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<32xf32> {
  %r = arith.addf %a, %b : vector<32xf32>
  return %r : vector<32xf32>
}

// -----

// Without contract the chain is not a MAC root, so the add stays legal.
// CHECK-LABEL: @macf_no_contract
// CHECK: arith.addf {{.*}} : vector<32xf32>
// CHECK-NOT: aievec.aie4.mac_ewise_f32
func.func @macf_no_contract(%a: vector<32xf32>, %b: vector<32xf32>,
                            %c: vector<32xf32>) -> vector<32xf32> {
  %m = arith.mulf %a, %b : vector<32xf32>
  %s = arith.addf %c, %m : vector<32xf32>
  return %s : vector<32xf32>
}

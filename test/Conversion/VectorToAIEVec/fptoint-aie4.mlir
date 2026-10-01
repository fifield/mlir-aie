//===- fptoint-aie4.mlir - AIE4 vector float -> int ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// AIE4 has a native float->int instruction covering <32xf32> -> <32xi32>, so
// that shape takes it. Everything else, and both int->float directions, has
// no instruction and scalarizes -- the prohibition on the rank-1 forms lives
// in configureAIEVecCommonLegalizations and applies to every target, so
// without a pattern AIE4 would have nothing to satisfy it.

// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie4b" | FileCheck %s

// The native shape: one op, no per-lane extraction.

// CHECK-LABEL: @fptosi_v32
// CHECK: aievec.aie4.conv_f32_i32 %{{.*}} : vector<32xf32>, vector<32xi32>
// CHECK-NOT: vector.extract
func.func @fptosi_v32(%a: vector<32xf32>) -> vector<32xi32> {
  %0 = arith.fptosi %a : vector<32xf32> to vector<32xi32>
  return %0 : vector<32xi32>
}

// -----

// CHECK-LABEL: @fptoui_v32
// CHECK: aievec.aie4.conv_f32_i32 %{{.*}} {isSigned = false} : vector<32xf32>, vector<32xi32>
// CHECK-NOT: vector.extract
func.func @fptoui_v32(%a: vector<32xf32>) -> vector<32xi32> {
  %0 = arith.fptoui %a : vector<32xf32> to vector<32xi32>
  return %0 : vector<32xi32>
}

// -----

// CHECK-LABEL: @fptosi_v16_scalarized
// CHECK-NOT: aievec.aie4.conv_f32_i32
// CHECK-COUNT-16: arith.fptosi %{{.*}} : f32 to i32
func.func @fptosi_v16_scalarized(%a: vector<16xf32>) -> vector<16xi32> {
  %0 = arith.fptosi %a : vector<16xf32> to vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

// Narrowing to i16 is a different instruction in the family and is not
// claimed here, so it scalarizes.

// CHECK-LABEL: @fptosi_i16_scalarized
// CHECK-NOT: aievec.aie4.conv_f32_i32
// CHECK: arith.fptosi %{{.*}} : f32 to i16
func.func @fptosi_i16_scalarized(%a: vector<32xf32>) -> vector<32xi16> {
  %0 = arith.fptosi %a : vector<32xf32> to vector<32xi16>
  return %0 : vector<32xi16>
}

// -----

// The other direction has no AIE4 instruction at all, so it scalarizes at the
// native width too.

// CHECK-LABEL: @sitofp_v32_scalarized
// CHECK-NOT: aievec.aie4.conv_f32_i32
// CHECK-COUNT-32: arith.sitofp %{{.*}} : i32 to f32
func.func @sitofp_v32_scalarized(%a: vector<32xi32>) -> vector<32xf32> {
  %0 = arith.sitofp %a : vector<32xi32> to vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

// A scalar conversion was never illegal and is left alone.

// CHECK-LABEL: @fptosi_scalar
// CHECK: arith.fptosi %{{.*}} : f32 to i32
// CHECK-NOT: aievec.aie4.conv_f32_i32
func.func @fptosi_scalar(%a: f32) -> i32 {
  %0 = arith.fptosi %a : f32 to i32
  return %0 : i32
}

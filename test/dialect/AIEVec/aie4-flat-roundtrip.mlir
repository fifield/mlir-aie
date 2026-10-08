// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s | FileCheck %s
// RUN: aie-opt %s | aie-opt | FileCheck %s
// RUN: aie-opt %s -mlir-print-op-generic | aie-opt | FileCheck %s
// RUN: aie-opt %s -mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC

// Flat matrix-state ops: stage, stage_stream, commit and mmac_active, for
// all four dense formats.

// CHECK-LABEL: func.func @stage
// CHECK: aievec.aie4.stage bf16 %{{.*}}, %{{.*}} : vector<64xbf16>, i32
// CHECK: aievec.aie4.stage int8 %{{.*}}, %{{.*}} {rhs_signed = true} : vector<128xi8>, i32
// CHECK: aievec.aie4.stage int4 %{{.*}}, %{{.*}} {rhs_signed = false} : vector<256xi4>, i32
// CHECK: aievec.aie4.stage int4 %{{.*}}, %{{.*}} {rhs_signed = true} : vector<128xi8>, i32
// CHECK: aievec.aie4.stage fp8 %{{.*}}, %{{.*}} : vector<128xf8E4M3FN>, i32
// CHECK: aievec.aie4.stage fp8 %{{.*}}, %{{.*}} : vector<128xi8>, i32
func.func @stage(%bf: vector<64xbf16>, %i8: vector<128xi8>, %i4: vector<256xi4>, %f8: vector<128xf8E4M3FN>, %slot: i32) {
  aievec.aie4.stage bf16 %bf, %slot : vector<64xbf16>, i32
  aievec.aie4.stage int8 %i8, %slot {rhs_signed = true} : vector<128xi8>, i32
  aievec.aie4.stage int4 %i4, %slot {rhs_signed = false} : vector<256xi4>, i32
  aievec.aie4.stage int4 %i8, %slot {rhs_signed = true} : vector<128xi8>, i32
  aievec.aie4.stage fp8 %f8, %slot : vector<128xf8E4M3FN>, i32
  aievec.aie4.stage fp8 %i8, %slot : vector<128xi8>, i32
  return
}

// Stream staging round-trips in custom and generic assembly.
// CHECK-LABEL: func.func @stage_stream
// CHECK: aievec.aie4.stage_stream bf16 %{{.*}}, %{{.*}} : i32, i32
// CHECK: aievec.aie4.stage_stream int8 %{{.*}}, %{{.*}} {rhs_signed = true} : i32, i32
// CHECK: aievec.aie4.stage_stream int4 %{{.*}}, %{{.*}} {rhs_signed = false} : i32, i32
// CHECK: aievec.aie4.stage_stream fp8 %{{.*}}, %{{.*}} : i32, i32
// GENERIC-LABEL: sym_name = "stage_stream"
// GENERIC: "aievec.aie4.stage_stream"(%{{.*}}, %{{.*}}) <{format = #aievec<aie4_matrix_format bf16>}> : (i32, i32) -> ()
// GENERIC: "aievec.aie4.stage_stream"(%{{.*}}, %{{.*}}) <{format = #aievec<aie4_matrix_format int8>, rhs_signed = true}> : (i32, i32) -> ()
func.func @stage_stream(%port: i32, %slot: i32) {
  aievec.aie4.stage_stream bf16 %port, %slot : i32, i32
  aievec.aie4.stage_stream int8 %port, %slot {rhs_signed = true} : i32, i32
  aievec.aie4.stage_stream int4 %port, %slot {rhs_signed = false} : i32, i32
  aievec.aie4.stage_stream fp8 %port, %slot : i32, i32
  return
}

// CHECK-LABEL: func.func @commit
// CHECK: aievec.aie4.commit bf16
// CHECK: aievec.aie4.commit int8 {rhs_signed = true}
// CHECK: aievec.aie4.commit int4 {rhs_signed = false}
// CHECK: aievec.aie4.commit fp8
func.func @commit() {
  aievec.aie4.commit bf16
  aievec.aie4.commit int8 {rhs_signed = true}
  aievec.aie4.commit int4 {rhs_signed = false}
  aievec.aie4.commit fp8
  return
}

// mmac_active with an accumulator, without one, and with zero_if. The
// zero_if operand type is implied (i1) and not printed.
// CHECK-LABEL: func.func @mmac_active
// CHECK: aievec.aie4.mmac_active bf16 %{{.*}}, %{{.*}} : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
// CHECK: aievec.aie4.mmac_active bf16 %{{.*}} : vector<32xbf16> -> vector<64xf32>
// CHECK: aievec.aie4.mmac_active int8 %{{.*}}, %{{.*}} {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
// CHECK: aievec.aie4.mmac_active int8 %{{.*}} {lhs_signed = true, rhs_signed = true} : vector<64xi8> -> vector<64xi32>
// CHECK: aievec.aie4.mmac_active int8 %{{.*}}, %{{.*}} zero_if %{{.*}} {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
// CHECK: aievec.aie4.mmac_active int4 %{{.*}}, %{{.*}} zero_if %{{.*}} {lhs_signed = false, rhs_signed = true} : vector<128xi4>, vector<64xi32> -> vector<64xi32>
// CHECK: aievec.aie4.mmac_active fp8 %{{.*}}, %{{.*}} : vector<64xf8E4M3FN>, vector<64xf32> -> vector<64xf32>
// CHECK: aievec.aie4.mmac_active fp8 %{{.*}} : vector<64xf8E4M3FN> -> vector<64xf32>
// GENERIC-LABEL: sym_name = "mmac_active"
// GENERIC: "aievec.aie4.mmac_active"(%{{.*}}, %{{.*}}) <{format = #aievec<aie4_matrix_format bf16>, operandSegmentSizes = array<i32: 1, 1, 0>}>
// GENERIC: "aievec.aie4.mmac_active"(%{{.*}}) <{format = #aievec<aie4_matrix_format bf16>, operandSegmentSizes = array<i32: 1, 0, 0>}>
// GENERIC: operandSegmentSizes = array<i32: 1, 1, 0>
// GENERIC: operandSegmentSizes = array<i32: 1, 0, 0>
// GENERIC: "aievec.aie4.mmac_active"(%{{.*}}, %{{.*}}, %{{.*}}) <{format = #aievec<aie4_matrix_format int8>, lhs_signed = true, operandSegmentSizes = array<i32: 1, 1, 1>, rhs_signed = true}> : (vector<64xi8>, vector<64xi32>, i1) -> vector<64xi32>
func.func @mmac_active(%abf: vector<32xbf16>, %cf: vector<64xf32>, %a8: vector<64xi8>, %a4: vector<128xi4>, %af8: vector<64xf8E4M3FN>, %ci: vector<64xi32>, %z: i1) {
  %0 = aievec.aie4.mmac_active bf16 %abf, %cf : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  %1 = aievec.aie4.mmac_active bf16 %abf : vector<32xbf16> -> vector<64xf32>
  %2 = aievec.aie4.mmac_active int8 %a8, %ci {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
  %3 = aievec.aie4.mmac_active int8 %a8 {lhs_signed = true, rhs_signed = true} : vector<64xi8> -> vector<64xi32>
  %4 = aievec.aie4.mmac_active int8 %a8, %ci zero_if %z {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
  %5 = aievec.aie4.mmac_active int4 %a4, %ci zero_if %z {lhs_signed = false, rhs_signed = true} : vector<128xi4>, vector<64xi32> -> vector<64xi32>
  %6 = aievec.aie4.mmac_active fp8 %af8, %cf : vector<64xf8E4M3FN>, vector<64xf32> -> vector<64xf32>
  %7 = aievec.aie4.mmac_active fp8 %af8 : vector<64xf8E4M3FN> -> vector<64xf32>
  return
}

// A flat sequence needs no enclosing region: stage a tile, commit it, then
// use it in a row loop.
// CHECK-LABEL: func.func @sequence
// CHECK: scf.for
// CHECK: aievec.aie4.stage int8
// CHECK: aievec.aie4.commit int8 {rhs_signed = true}
// CHECK: scf.for
// CHECK: aievec.aie4.mmac_active int8
func.func @sequence(%chunk: vector<128xi8>, %a: vector<64xi8>, %c: vector<64xi32>) -> vector<64xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  scf.for %s = %c0 to %c32 step %c1 {
    %slot = arith.index_cast %s : index to i32
    aievec.aie4.stage int8 %chunk, %slot {rhs_signed = true} : vector<128xi8>, i32
  }
  aievec.aie4.commit int8 {rhs_signed = true}
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c) -> (vector<64xi32>) {
    %n = aievec.aie4.mmac_active int8 %a, %acc {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
    scf.yield %n : vector<64xi32>
  }
  return %r : vector<64xi32>
}

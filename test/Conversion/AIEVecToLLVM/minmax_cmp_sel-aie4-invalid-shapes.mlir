// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Native operations and intrinsics require fixed rank-one vectors.

func.func @minmax_scalable(%a: vector<[16]xi32>, %m: ui32) {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %r = "aievec.aie4.minmax"(%a, %a) {kind = #aievec.aie4_minmax_kind<min>, isSigned = true} : (vector<[16]xi32>, vector<[16]xi32>) -> vector<[16]xi32>
  return
}

// -----

func.func @cmp_scalable(%a: vector<[16]xi32>, %m: ui32) {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %r = "aievec.aie4.cmp"(%a, %a) {predicate = #aievec.aie4_cmp_predicate<lt>, isSigned = true} : (vector<[16]xi32>, vector<[16]xi32>) -> ui32
  return
}

// -----

func.func @sel_scalable(%a: vector<[16]xi32>, %m: ui32) {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %r = "aievec.aie4.sel"(%a, %a, %m)  : (vector<[16]xi32>, vector<[16]xi32>, ui32) -> vector<[16]xi32>
  return
}

// -----

func.func @intrinsic_vadd32_512_scalable(%a: vector<[16]xi32>, %m: i32) {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %r = "xllvm.intr.aie4.vadd32.512"(%a, %a) {opsel = 15 : i32, sign = 1 : i32} : (vector<[16]xi32>, vector<[16]xi32>) -> !llvm.struct<(vector<16xi32>, i32)>
  return
}

// -----

func.func @intrinsic_vadd32_mask_512_scalable(%a: vector<[16]xi32>, %m: i32) {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %r = "xllvm.intr.aie4.vadd32.mask.512"(%a, %a, %m)  : (vector<[16]xi32>, vector<[16]xi32>, i32) -> vector<16xi32>
  return
}

// -----

func.func @minmax_rank_two(%a: vector<4x4xi32>, %m: ui32) {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %r = "aievec.aie4.minmax"(%a, %a) {kind = #aievec.aie4_minmax_kind<min>, isSigned = true} : (vector<4x4xi32>, vector<4x4xi32>) -> vector<4x4xi32>
  return
}

// -----

func.func @cmp_rank_two(%a: vector<4x4xi32>, %m: ui32) {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %r = "aievec.aie4.cmp"(%a, %a) {predicate = #aievec.aie4_cmp_predicate<lt>, isSigned = true} : (vector<4x4xi32>, vector<4x4xi32>) -> ui32
  return
}

// -----

func.func @sel_rank_two(%a: vector<4x4xi32>, %m: ui32) {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %r = "aievec.aie4.sel"(%a, %a, %m)  : (vector<4x4xi32>, vector<4x4xi32>, ui32) -> vector<4x4xi32>
  return
}

// -----

func.func @intrinsic_vadd32_512_rank_two(%a: vector<4x4xi32>, %m: i32) {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %r = "xllvm.intr.aie4.vadd32.512"(%a, %a) {opsel = 15 : i32, sign = 1 : i32} : (vector<4x4xi32>, vector<4x4xi32>) -> !llvm.struct<(vector<16xi32>, i32)>
  return
}

// -----

func.func @intrinsic_vadd32_mask_512_rank_two(%a: vector<4x4xi32>, %m: i32) {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %r = "xllvm.intr.aie4.vadd32.mask.512"(%a, %a, %m)  : (vector<4x4xi32>, vector<4x4xi32>, i32) -> vector<16xi32>
  return
}

// -----

func.func @intrinsic_scalable_mask(%a: vector<64xi8>, %m: vector<[2]xi32>) {
  // expected-error @+1 {{operand #2 must be fixed-length vector}}
  %r = "xllvm.intr.aie4.vadd8.mask.512"(%a, %a, %m) : (vector<64xi8>, vector<64xi8>, vector<[2]xi32>) -> vector<64xi8>
  return
}

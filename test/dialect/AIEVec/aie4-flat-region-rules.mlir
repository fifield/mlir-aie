// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Region bodies reject flat writers and use mmac instead of mmac_active.
// See aie4-mmac-rules.mlir for the remaining region constraints.

func.func @stage_in_region(%b: vector<64x64xi8>, %chunk: vector<128xi8>, %i: i32) {
  // expected-error @+1 {{body must not contain matrix staging or commit operations; it may replace or clobber the active matrix}}
  aievec.aie4.matrix_region int8 %b : vector<64x64xi8> attributes {rhs_signed = true} {
    // expected-note @+1 {{matrix-state write here}}
    aievec.aie4.stage int8 %chunk, %i {rhs_signed = true} : vector<128xi8>, i32
    aievec.aie4.matrix_yield
  }
  return
}

// -----

func.func @stage_stream_in_region_loop(%b: vector<32x64xbf16>, %p: i32, %i: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // expected-error @+1 {{body must not contain matrix staging or commit operations}}
  aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> {
    scf.for %k = %c0 to %c4 step %c1 {
      // expected-note @+1 {{matrix-state write here}}
      aievec.aie4.stage_stream bf16 %p, %i : i32, i32
    }
    aievec.aie4.matrix_yield
  }
  return
}

// -----

func.func @commit_in_region(%b: vector<64x64xf8E4M3FN>) {
  // expected-error @+1 {{body must not contain matrix staging or commit operations}}
  aievec.aie4.matrix_region fp8 %b : vector<64x64xf8E4M3FN> {
    // expected-note @+1 {{matrix-state write here}}
    aievec.aie4.commit fp8
    aievec.aie4.matrix_yield
  }
  return
}

// -----

func.func @mmac_active_in_region(%b: vector<32x64xbf16>, %a: vector<32xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  // expected-error @+1 {{body must not contain 'aievec.aie4.mmac_active'; region bodies use 'aievec.aie4.mmac'}}
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> -> vector<64xf32> {
    // expected-note @+1 {{flat MMAC here}}
    %x = aievec.aie4.mmac_active bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// -----

func.func @mmac_active_in_region_loop(%b: vector<64x64xi8>, %a: vector<64xi8>, %c: vector<64xi32>) -> vector<64xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // expected-error @+1 {{body must not contain 'aievec.aie4.mmac_active'}}
  %r = aievec.aie4.matrix_region int8 %b : vector<64x64xi8>, %c : vector<64xi32> -> vector<64xi32> attributes {rhs_signed = true} {
  ^bb0(%acc: vector<64xi32>):
    %n = scf.for %i = %c0 to %c4 step %c1 iter_args(%x = %acc) -> (vector<64xi32>) {
      // expected-note @+1 {{flat MMAC here}}
      %y = aievec.aie4.mmac_active int8 %a, %x {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
      scf.yield %y : vector<64xi32>
    }
    aievec.aie4.matrix_yield %n : vector<64xi32>
  }
  return %r : vector<64xi32>
}

// -----

//===----------------------------------------------------------------------===//
// Accepted forms
//===----------------------------------------------------------------------===//

// Raw XLLVM MMACs read the matrix and remain allowed in a body.
func.func @xllvm_int_mmac_in_region(%b: vector<64x64xi8>, %a: vector<64xi8>, %c: vector<64xi32>, %conf: i32) -> vector<64xi32> {
  %r = aievec.aie4.matrix_region int8 %b : vector<64x64xi8> -> vector<64xi32> attributes {rhs_signed = true} {
    %x = "xllvm.intr.aie4.I512.ACC2048.mmac"(%c, %a) {bmode = 1 : i32, xSign = 1 : i32, zSign = 1 : i32} : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
    %y = "xllvm.intr.aie4.I512.ACC2048.mmac.dyn_conf"(%x, %a, %conf) : (vector<64xi32>, vector<64xi8>, i32) -> vector<64xi32>
    aievec.aie4.matrix_yield %y : vector<64xi32>
  }
  return %r : vector<64xi32>
}

// -----

// Flat ops before and after a region are fine.
func.func @flat_around_region(%b: vector<32x64xbf16>, %chunk: vector<64xbf16>, %i: i32, %a: vector<32xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  aievec.aie4.stage bf16 %chunk, %i : vector<64xbf16>, i32
  aievec.aie4.commit bf16
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> -> vector<64xf32> {
    %x = aievec.aie4.mmac bf16 %a, %c : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  %y = aievec.aie4.mmac_active bf16 %a, %r : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  return %y : vector<64xf32>
}

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" -verify-diagnostics

// A row-major B keeps the staging transpose, which lowers element by
// element. The lowering warns; a column-major source or a splat does not.

func.func @row_major(%a: vector<32xbf16>, %b: vector<32x64xbf16>) -> vector<64xf32> {
  // expected-warning @+1 {{B tile is not produced by a vector.transpose [1, 0] of a column-major tile; staging transposes it element by element (about 44x the staging cost for one bf16 tile). Supply B column-major so the transpose cancels}}
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> -> vector<64xf32> {
    %x = aievec.aie4.mmac bf16 %a : vector<32xbf16> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// -----

// The semantic op warns through its region.
func.func @matmul_row_major(%a: vector<2x64xi8>, %b: vector<64x64xi8>, %c: vector<2x64xi32>) -> vector<2x64xi32> {
  // expected-warning @+1 {{B tile is not produced by a vector.transpose [1, 0]}}
  %0 = aievec.aie4.matmul %a, %b, %c {lhs_signed = true, rhs_signed = true} : vector<2x64xi8>, vector<64x64xi8> into vector<2x64xi32>
  return %0 : vector<2x64xi32>
}

// -----

func.func @column_major(%a: vector<32xbf16>, %bt: vector<64x32xbf16>) -> vector<64xf32> {
  %b = vector.transpose %bt, [1, 0] : vector<64x32xbf16> to vector<32x64xbf16>
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> -> vector<64xf32> {
    %x = aievec.aie4.mmac bf16 %a : vector<32xbf16> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// -----

func.func @splat(%a: vector<64xi8>, %s: i8) -> vector<64xi32> {
  %b = vector.broadcast %s : i8 to vector<64x64xi8>
  %r = aievec.aie4.matrix_region int8 %b : vector<64x64xi8> -> vector<64xi32> attributes {rhs_signed = true} {
    %x = aievec.aie4.mmac int8 %a {lhs_signed = true, rhs_signed = true} : vector<64xi8> -> vector<64xi32>
    aievec.aie4.matrix_yield %x : vector<64xi32>
  }
  return %r : vector<64xi32>
}

// -----

// Only a [1, 0] transpose cancels the staging transpose.
func.func @identity_transpose(%a: vector<32xbf16>, %b0: vector<32x64xbf16>) -> vector<64xf32> {
  %b = vector.transpose %b0, [0, 1] : vector<32x64xbf16> to vector<32x64xbf16>
  // expected-warning @+1 {{B tile is not produced by a vector.transpose [1, 0]}}
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> -> vector<64xf32> {
    %x = aievec.aie4.mmac bf16 %a : vector<32xbf16> -> vector<64xf32>
    aievec.aie4.matrix_yield %x : vector<64xf32>
  }
  return %r : vector<64xf32>
}

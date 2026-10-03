// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Shape, element type, format and signedness checks for the AIE4 MMAC ops.

//===----------------------------------------------------------------------===//
// aievec.aie4.matmul
//===----------------------------------------------------------------------===//

func.func @matmul_bf16_wrong_k(%a: vector<4x64xbf16>, %b: vector<32x64xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  // expected-error @+1 {{lhs must have 32 columns for 'bf16', got 'vector<4x64xbf16>'}}
  %0 = aievec.aie4.matmul %a, %b, %c : vector<4x64xbf16>, vector<32x64xbf16> into vector<4x64xf32>
  return %0 : vector<4x64xf32>
}

// -----

// Larger K must be tiled by the producer: rhs is exactly one B tile.
func.func @matmul_bf16_two_k_tiles(%a: vector<4x64xbf16>, %b: vector<64x64xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  // expected-error @+1 {{rhs must be exactly one native B tile 'vector<32x64xbf16>', got 'vector<64x64xbf16>'}}
  %0 = aievec.aie4.matmul %a, %b, %c : vector<4x64xbf16>, vector<64x64xbf16> into vector<4x64xf32>
  return %0 : vector<4x64xf32>
}

// -----

func.func @matmul_i8_wide_n(%a: vector<4x64xi8>, %b: vector<64x128xi8>, %c: vector<4x128xi32>) -> vector<4x128xi32> {
  // expected-error @+1 {{rhs must be exactly one native B tile 'vector<64x64xi8>', got 'vector<64x128xi8>'}}
  %0 = aievec.aie4.matmul %a, %b, %c {lhs_signed = true, rhs_signed = true} : vector<4x64xi8>, vector<64x128xi8> into vector<4x128xi32>
  return %0 : vector<4x128xi32>
}

// -----

func.func @matmul_i4_as_i8_k(%a: vector<4x64xi4>, %b: vector<128x64xi4>, %c: vector<4x64xi32>) -> vector<4x64xi32> {
  // expected-error @+1 {{lhs must have 128 columns for 'i4', got 'vector<4x64xi4>'}}
  %0 = aievec.aie4.matmul %a, %b, %c {lhs_signed = true, rhs_signed = true} : vector<4x64xi4>, vector<128x64xi4> into vector<4x64xi32>
  return %0 : vector<4x64xi32>
}

// -----

func.func @matmul_bf16_int_acc(%a: vector<4x32xbf16>, %b: vector<32x64xbf16>, %c: vector<4x64xi32>) -> vector<4x64xi32> {
  // expected-error @+1 {{acc must be 'vector<4x64xf32>' for lhs 'vector<4x32xbf16>', got 'vector<4x64xi32>'}}
  %0 = aievec.aie4.matmul %a, %b, %c : vector<4x32xbf16>, vector<32x64xbf16> into vector<4x64xi32>
  return %0 : vector<4x64xi32>
}

// -----

func.func @matmul_acc_rows_mismatch(%a: vector<4x32xbf16>, %b: vector<32x64xbf16>, %c: vector<8x64xf32>) -> vector<8x64xf32> {
  // expected-error @+1 {{acc must be 'vector<4x64xf32>' for lhs 'vector<4x32xbf16>', got 'vector<8x64xf32>'}}
  %0 = aievec.aie4.matmul %a, %b, %c : vector<4x32xbf16>, vector<32x64xbf16> into vector<8x64xf32>
  return %0 : vector<8x64xf32>
}

// -----

func.func @matmul_mixed_elements(%a: vector<4x64xi8>, %b: vector<64x64xf8E4M3FN>, %c: vector<4x64xi32>) -> vector<4x64xi32> {
  // expected-error @+1 {{rhs element type 'f8E4M3FN' must match lhs element type 'i8'}}
  %0 = aievec.aie4.matmul %a, %b, %c {lhs_signed = true, rhs_signed = true} : vector<4x64xi8>, vector<64x64xf8E4M3FN> into vector<4x64xi32>
  return %0 : vector<4x64xi32>
}

// -----

// Only the OCP E4M3 fp8 variant exists on AIE4.
func.func @matmul_fp8_e5m2(%a: vector<4x64xf8E5M2>, %b: vector<64x64xf8E5M2>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  // expected-error @+1 {{unsupported lhs element type 'f8E5M2'; expected bf16, i8, i4 or f8E4M3FN}}
  %0 = aievec.aie4.matmul %a, %b, %c : vector<4x64xf8E5M2>, vector<64x64xf8E5M2> into vector<4x64xf32>
  return %0 : vector<4x64xf32>
}

// -----

func.func @matmul_f16(%a: vector<4x32xf16>, %b: vector<32x64xf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  // expected-error @+1 {{unsupported lhs element type 'f16'}}
  %0 = aievec.aie4.matmul %a, %b, %c : vector<4x32xf16>, vector<32x64xf16> into vector<4x64xf32>
  return %0 : vector<4x64xf32>
}

// -----

func.func @matmul_rank1_lhs(%a: vector<32xbf16>, %b: vector<32x64xbf16>, %c: vector<1x64xf32>) -> vector<1x64xf32> {
  // expected-error @+1 {{lhs, rhs and acc must be rank-2 vectors}}
  %0 = aievec.aie4.matmul %a, %b, %c : vector<32xbf16>, vector<32x64xbf16> into vector<1x64xf32>
  return %0 : vector<1x64xf32>
}

// -----

func.func @matmul_scalable(%a: vector<[4]x32xbf16>, %b: vector<32x64xbf16>, %c: vector<[4]x64xf32>) -> vector<[4]x64xf32> {
  // expected-error @+1 {{operands must be fixed-length vectors}}
  %0 = aievec.aie4.matmul %a, %b, %c : vector<[4]x32xbf16>, vector<32x64xbf16> into vector<[4]x64xf32>
  return %0 : vector<[4]x64xf32>
}

// -----

func.func @matmul_int_missing_lhs_sign(%a: vector<4x64xi8>, %b: vector<64x64xi8>, %c: vector<4x64xi32>) -> vector<4x64xi32> {
  // expected-error @+1 {{integer format requires 'lhs_signed'}}
  %0 = aievec.aie4.matmul %a, %b, %c {rhs_signed = true} : vector<4x64xi8>, vector<64x64xi8> into vector<4x64xi32>
  return %0 : vector<4x64xi32>
}

// -----

func.func @matmul_int_missing_rhs_sign(%a: vector<4x64xi8>, %b: vector<64x64xi8>, %c: vector<4x64xi32>) -> vector<4x64xi32> {
  // expected-error @+1 {{integer format requires 'rhs_signed'}}
  %0 = aievec.aie4.matmul %a, %b, %c {lhs_signed = true} : vector<4x64xi8>, vector<64x64xi8> into vector<4x64xi32>
  return %0 : vector<4x64xi32>
}

// -----

func.func @matmul_float_with_sign(%a: vector<4x32xbf16>, %b: vector<32x64xbf16>, %c: vector<4x64xf32>) -> vector<4x64xf32> {
  // expected-error @+1 {{'lhs_signed' applies only to integer formats}}
  %0 = aievec.aie4.matmul %a, %b, %c {lhs_signed = true} : vector<4x32xbf16>, vector<32x64xbf16> into vector<4x64xf32>
  return %0 : vector<4x64xf32>
}

// -----

//===----------------------------------------------------------------------===//
// aievec.aie4.matrix_region
//===----------------------------------------------------------------------===//

// The format attribute must agree with the B tile type.
func.func @region_format_vs_tile(%b: vector<64x64xi8>) {
  // expected-error @+1 {{rhs must be the bf16 B tile 'vector<32x64xbf16>', got 'vector<64x64xi8>'}}
  aievec.aie4.matrix_region bf16 %b : vector<64x64xi8> {
    aievec.aie4.matrix_yield
  }
  return
}

// -----

func.func @region_int4_packed_tile(%b: vector<64x64xi8>) {
  // expected-error @+1 {{rhs must be the int4 B tile 'vector<128x64xi4>', got 'vector<64x64xi8>'}}
  aievec.aie4.matrix_region int4 %b : vector<64x64xi8> attributes {rhs_signed = true} {
    aievec.aie4.matrix_yield
  }
  return
}

// -----

func.func @region_int_missing_sign(%b: vector<64x64xi8>) {
  // expected-error @+1 {{integer format requires 'rhs_signed'}}
  aievec.aie4.matrix_region int8 %b : vector<64x64xi8> {
    aievec.aie4.matrix_yield
  }
  return
}

// -----

func.func @region_fp8_with_sign(%b: vector<64x64xf8E4M3FN>) {
  // expected-error @+1 {{'rhs_signed' applies only to integer formats}}
  aievec.aie4.matrix_region fp8 %b : vector<64x64xf8E4M3FN> attributes {rhs_signed = true} {
    aievec.aie4.matrix_yield
  }
  return
}

// -----

func.func @region_block_args(%b: vector<32x64xbf16>, %c: vector<64xf32>) {
  // expected-error @+1 {{entry block argument types must match the types of 'inits'}}
  aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16>, %c : vector<64xf32> {
    aievec.aie4.matrix_yield
  }
  return
}

// -----

func.func @region_yield_types(%b: vector<32x64xbf16>, %c: vector<64xf32>) -> vector<64xf32> {
  // expected-error @+1 {{yielded types must match the result types}}
  %r = aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16>, %c : vector<64xf32> -> vector<64xf32> {
  ^bb0(%acc: vector<64xf32>):
    aievec.aie4.matrix_yield
  }
  return %r : vector<64xf32>
}

// -----

func.func @yield_outside_region() {
  // expected-error @+1 {{expects parent op 'aievec.aie4.matrix_region'}}
  aievec.aie4.matrix_yield
}

// -----

//===----------------------------------------------------------------------===//
// aievec.aie4.mmac
//===----------------------------------------------------------------------===//

// AIEVec int4 rows use logical i4 elements; packing to i8 happens in lowering.
func.func @mmac_int4_packed_row(%b: vector<128x64xi4>, %a: vector<64xi8>) {
  aievec.aie4.matrix_region int4 %b : vector<128x64xi4> attributes {rhs_signed = true} {
    // expected-error @+1 {{lhs must be 'vector<128xi4>' for format int4, got 'vector<64xi8>'}}
    %r = aievec.aie4.mmac int4 %a {lhs_signed = true, rhs_signed = true} : vector<64xi8> -> vector<64xi32>
    aievec.aie4.matrix_yield
  }
  return
}

// -----

func.func @mmac_bf16_wrong_row(%b: vector<32x64xbf16>, %a: vector<64xbf16>) {
  aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> {
    // expected-error @+1 {{lhs must be 'vector<32xbf16>' for format bf16, got 'vector<64xbf16>'}}
    %r = aievec.aie4.mmac bf16 %a : vector<64xbf16> -> vector<64xf32>
    aievec.aie4.matrix_yield
  }
  return
}

// -----

// Each MMAC returns one 64-lane row, not a block of accumulator rows.
func.func @mmac_block_acc(%b: vector<32x64xbf16>, %a: vector<32xbf16>, %c: vector<32x64xf32>) {
  aievec.aie4.matrix_region bf16 %b : vector<32x64xbf16> {
    // expected-error @+1 {{acc must be 'vector<64xf32>' for format bf16, got 'vector<32x64xf32>'}}
    %r = aievec.aie4.mmac bf16 %a, %c : vector<32xbf16>, vector<32x64xf32> -> vector<64xf32>
    aievec.aie4.matrix_yield
  }
  return
}

// -----

func.func @mmac_int8_float_result(%b: vector<64x64xi8>, %a: vector<64xi8>) {
  aievec.aie4.matrix_region int8 %b : vector<64x64xi8> attributes {rhs_signed = false} {
    // expected-error @+1 {{result must be 'vector<64xi32>' for format int8, got 'vector<64xf32>'}}
    %r = aievec.aie4.mmac int8 %a {lhs_signed = false, rhs_signed = false} : vector<64xi8> -> vector<64xf32>
    aievec.aie4.matrix_yield
  }
  return
}

// -----

func.func @mmac_int_missing_lhs_sign(%b: vector<64x64xi8>, %a: vector<64xi8>) {
  aievec.aie4.matrix_region int8 %b : vector<64x64xi8> attributes {rhs_signed = false} {
    // expected-error @+1 {{integer format requires 'lhs_signed'}}
    %r = aievec.aie4.mmac int8 %a {rhs_signed = false} : vector<64xi8> -> vector<64xi32>
    aievec.aie4.matrix_yield
  }
  return
}

// -----

func.func @mmac_float_with_sign(%b: vector<64x64xf8E4M3FN>, %a: vector<64xf8E4M3FN>) {
  aievec.aie4.matrix_region fp8 %b : vector<64x64xf8E4M3FN> {
    // expected-error @+1 {{'rhs_signed' applies only to integer formats}}
    %r = aievec.aie4.mmac fp8 %a {rhs_signed = true} : vector<64xf8E4M3FN> -> vector<64xf32>
    aievec.aie4.matrix_yield
  }
  return
}

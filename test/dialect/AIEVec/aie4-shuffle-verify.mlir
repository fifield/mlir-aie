// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Accept all four modes and reject mismatched element widths, operand
// counts, ranks, and scalable vectors.

// RUN: aie-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @shuffle_t32_4x4
func.func @shuffle_t32_4x4(%a : vector<16xi32>) -> vector<16xi32> {
  // CHECK: aievec.aie4.shuffle %{{.*}}[mode = <t32_4x4>] : vector<16xi32>
  %0 = aievec.aie4.shuffle %a [mode = <t32_4x4>] : vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

// CHECK-LABEL: @shuffle_t8_8x8
func.func @shuffle_t8_8x8(%a : vector<64xi8>) -> vector<64xi8> {
  // CHECK: aievec.aie4.shuffle %{{.*}}[mode = <t8_8x8>] : vector<64xi8>
  %0 = aievec.aie4.shuffle %a [mode = <t8_8x8>] : vector<64xi8>
  return %0 : vector<64xi8>
}

// -----

// CHECK-LABEL: @shuffle_t32_2x16_lo
func.func @shuffle_t32_2x16_lo(%a : vector<16xi32>, %b : vector<16xi32>) -> vector<16xi32> {
  // CHECK: aievec.aie4.shuffle %{{.*}}, %{{.*}}[mode = <t32_2x16_lo>] : vector<16xi32>
  %0 = aievec.aie4.shuffle %a, %b [mode = <t32_2x16_lo>] : vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

// CHECK-LABEL: @shuffle_t32_2x16_hi
func.func @shuffle_t32_2x16_hi(%a : vector<16xi32>, %b : vector<16xi32>) -> vector<16xi32> {
  // CHECK: aievec.aie4.shuffle %{{.*}}, %{{.*}}[mode = <t32_2x16_hi>] : vector<16xi32>
  %0 = aievec.aie4.shuffle %a, %b [mode = <t32_2x16_hi>] : vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

func.func @shuffle_lo_missing_rhs(%a : vector<16xi32>) -> vector<16xi32> {
  // expected-error@+1 {{shuffle mode 't32_2x16_lo' requires a second operand}}
  %0 = aievec.aie4.shuffle %a [mode = <t32_2x16_lo>] : vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

func.func @shuffle_4x4_extra_rhs(%a : vector<16xi32>, %b : vector<16xi32>) -> vector<16xi32> {
  // expected-error@+1 {{shuffle mode 't32_4x4' does not admit a second operand}}
  %0 = aievec.aie4.shuffle %a, %b [mode = <t32_4x4>] : vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

func.func @shuffle_4x4_wrong_width(%a : vector<64xi8>) -> vector<64xi8> {
  // expected-error@+1 {{shuffle mode 't32_4x4' requires vectors of 32-bit elements}}
  %0 = aievec.aie4.shuffle %a [mode = <t32_4x4>] : vector<64xi8>
  return %0 : vector<64xi8>
}

// -----

func.func @shuffle_8x8_wrong_width(%a : vector<16xi32>) -> vector<16xi32> {
  // expected-error@+1 {{shuffle mode 't8_8x8' requires vectors of 8-bit elements}}
  %0 = aievec.aie4.shuffle %a [mode = <t8_8x8>] : vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

func.func @shuffle_rank_two_lhs(%a: vector<4x4xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // expected-error@+1 {{operand #0 must be fixed one-dimensional 512-bit vector}}
  %0 = "aievec.aie4.shuffle"(%a, %b) <{mode = #aievec.aie4_shuffle_mode<t32_2x16_lo>}> : (vector<4x4xi32>, vector<16xi32>) -> vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

func.func @shuffle_rank_two_rhs(%a: vector<16xi32>, %b: vector<4x4xi32>) -> vector<16xi32> {
  // expected-error@+1 {{operand #1 must be fixed one-dimensional 512-bit vector}}
  %0 = "aievec.aie4.shuffle"(%a, %b) <{mode = #aievec.aie4_shuffle_mode<t32_2x16_lo>}> : (vector<16xi32>, vector<4x4xi32>) -> vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

func.func @shuffle_rank_two_result(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<4x4xi32> {
  // expected-error@+1 {{result #0 must be fixed one-dimensional 512-bit vector}}
  %0 = "aievec.aie4.shuffle"(%a, %b) <{mode = #aievec.aie4_shuffle_mode<t32_2x16_lo>}> : (vector<16xi32>, vector<16xi32>) -> vector<4x4xi32>
  return %0 : vector<4x4xi32>
}

// -----

func.func @shuffle_scalable_lhs(%a: vector<[16]xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // expected-error@+1 {{operand #0 must be fixed one-dimensional 512-bit vector}}
  %0 = "aievec.aie4.shuffle"(%a, %b) <{mode = #aievec.aie4_shuffle_mode<t32_2x16_lo>}> : (vector<[16]xi32>, vector<16xi32>) -> vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

func.func @shuffle_scalable_rhs(%a: vector<16xi32>, %b: vector<[16]xi32>) -> vector<16xi32> {
  // expected-error@+1 {{operand #1 must be fixed one-dimensional 512-bit vector}}
  %0 = "aievec.aie4.shuffle"(%a, %b) <{mode = #aievec.aie4_shuffle_mode<t32_2x16_lo>}> : (vector<16xi32>, vector<[16]xi32>) -> vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

func.func @shuffle_scalable_result(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<[16]xi32> {
  // expected-error@+1 {{result #0 must be fixed one-dimensional 512-bit vector}}
  %0 = "aievec.aie4.shuffle"(%a, %b) <{mode = #aievec.aie4_shuffle_mode<t32_2x16_lo>}> : (vector<16xi32>, vector<16xi32>) -> vector<[16]xi32>
  return %0 : vector<[16]xi32>
}

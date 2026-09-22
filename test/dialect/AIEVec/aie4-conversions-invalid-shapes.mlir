// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// -----

func.func @ups_i16_i32_scalable_source(%a: vector<[32]xi16>) -> vector<32xi32> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %r = "aievec.aie4.ups_i16_i32"(%a) {shift = 0 : i32} : (vector<[32]xi16>) -> vector<32xi32>
  return %r : vector<32xi32>
}

// -----

func.func @ups_i16_i32_scalable_result(%a: vector<32xi16>) -> vector<[32]xi32> {
  // expected-error @+1 {{result #0 must be fixed-length vector}}
  %r = "aievec.aie4.ups_i16_i32"(%a) {shift = 0 : i32} : (vector<32xi16>) -> vector<[32]xi32>
  return %r : vector<[32]xi32>
}

// -----

func.func @ups_i16_i32_rank_two_source(%a: vector<4x8xi16>) -> vector<32xi32> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %r = "aievec.aie4.ups_i16_i32"(%a) {shift = 0 : i32} : (vector<4x8xi16>) -> vector<32xi32>
  return %r : vector<32xi32>
}

// -----

func.func @ups_i16_i32_rank_two_result(%a: vector<32xi16>) -> vector<4x8xi32> {
  // expected-error @+1 {{result #0 must be fixed-length vector}}
  %r = "aievec.aie4.ups_i16_i32"(%a) {shift = 0 : i32} : (vector<32xi16>) -> vector<4x8xi32>
  return %r : vector<4x8xi32>
}

// -----

func.func @conv_bf16_f32_scalable_source(%a: vector<[32]xbf16>) -> vector<32xf32> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %r = "aievec.aie4.conv_bf16_f32"(%a) : (vector<[32]xbf16>) -> vector<32xf32>
  return %r : vector<32xf32>
}

// -----

func.func @conv_bf16_f32_scalable_result(%a: vector<32xbf16>) -> vector<[32]xf32> {
  // expected-error @+1 {{result #0 must be fixed-length vector}}
  %r = "aievec.aie4.conv_bf16_f32"(%a) : (vector<32xbf16>) -> vector<[32]xf32>
  return %r : vector<[32]xf32>
}

// -----

func.func @conv_bf16_f32_rank_two_source(%a: vector<4x8xbf16>) -> vector<32xf32> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %r = "aievec.aie4.conv_bf16_f32"(%a) : (vector<4x8xbf16>) -> vector<32xf32>
  return %r : vector<32xf32>
}

// -----

func.func @conv_bf16_f32_rank_two_result(%a: vector<32xbf16>) -> vector<4x8xf32> {
  // expected-error @+1 {{result #0 must be fixed-length vector}}
  %r = "aievec.aie4.conv_bf16_f32"(%a) : (vector<32xbf16>) -> vector<4x8xf32>
  return %r : vector<4x8xf32>
}

// -----

func.func @conv_f32_bf16_scalable_source(%a: vector<[32]xf32>) -> vector<32xbf16> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %r = "aievec.aie4.conv_f32_bf16"(%a) : (vector<[32]xf32>) -> vector<32xbf16>
  return %r : vector<32xbf16>
}

// -----

func.func @conv_f32_bf16_scalable_result(%a: vector<32xf32>) -> vector<[32]xbf16> {
  // expected-error @+1 {{result #0 must be fixed-length vector}}
  %r = "aievec.aie4.conv_f32_bf16"(%a) : (vector<32xf32>) -> vector<[32]xbf16>
  return %r : vector<[32]xbf16>
}

// -----

func.func @conv_f32_bf16_rank_two_source(%a: vector<4x8xf32>) -> vector<32xbf16> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %r = "aievec.aie4.conv_f32_bf16"(%a) : (vector<4x8xf32>) -> vector<32xbf16>
  return %r : vector<32xbf16>
}

// -----

func.func @conv_f32_bf16_rank_two_result(%a: vector<32xf32>) -> vector<4x8xbf16> {
  // expected-error @+1 {{result #0 must be fixed-length vector}}
  %r = "aievec.aie4.conv_f32_bf16"(%a) : (vector<32xf32>) -> vector<4x8xbf16>
  return %r : vector<4x8xbf16>
}

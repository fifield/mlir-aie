// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics

func.func @fminmax_scalable_operand0(%a: vector<[32]xbf16>, %b: vector<32xbf16>) -> vector<32xbf16> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "aievec.aie4.fminmax"(%a, %b) {kind = #aievec.aie4_minmax_kind<min>} : (vector<[32]xbf16>, vector<32xbf16>) -> vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

func.func @fminmax_scalable_operand1(%a: vector<64xf16>, %b: vector<[64]xf16>) -> vector<64xf16> {
  // expected-error @+1 {{operand #1 must be fixed-length vector}}
  %0 = "aievec.aie4.fminmax"(%a, %b) {kind = #aievec.aie4_minmax_kind<min>} : (vector<64xf16>, vector<[64]xf16>) -> vector<64xf16>
  return %0 : vector<64xf16>
}

// -----

func.func @fminmax_scalable_result(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<[16]xf32> {
  // expected-error @+1 {{result #0 must be fixed-length vector}}
  %0 = "aievec.aie4.fminmax"(%a, %b) {kind = #aievec.aie4_minmax_kind<min>} : (vector<16xf32>, vector<16xf32>) -> vector<[16]xf32>
  return %0 : vector<[16]xf32>
}

// -----

func.func @fminmax_rank_two_operand0(%a: vector<2x16xbf16>, %b: vector<32xbf16>) -> vector<32xbf16> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "aievec.aie4.fminmax"(%a, %b) {kind = #aievec.aie4_minmax_kind<min>} : (vector<2x16xbf16>, vector<32xbf16>) -> vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

func.func @fminmax_rank_two_operand1(%a: vector<64xf16>, %b: vector<2x32xf16>) -> vector<64xf16> {
  // expected-error @+1 {{operand #1 must be fixed-length vector}}
  %0 = "aievec.aie4.fminmax"(%a, %b) {kind = #aievec.aie4_minmax_kind<min>} : (vector<64xf16>, vector<2x32xf16>) -> vector<64xf16>
  return %0 : vector<64xf16>
}

// -----

func.func @fminmax_rank_two_result(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<2x8xf32> {
  // expected-error @+1 {{result #0 must be fixed-length vector}}
  %0 = "aievec.aie4.fminmax"(%a, %b) {kind = #aievec.aie4_minmax_kind<min>} : (vector<16xf32>, vector<16xf32>) -> vector<2x8xf32>
  return %0 : vector<2x8xf32>
}

// -----

func.func @fcmp_scalable_operand0(%a: vector<[32]xbf16>, %b: vector<32xbf16>) -> ui32 {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "aievec.aie4.fcmp"(%a, %b) {predicate = #aievec.aie4_cmp_predicate<lt>} : (vector<[32]xbf16>, vector<32xbf16>) -> ui32
  return %0 : ui32
}

// -----

func.func @fcmp_scalable_operand1(%a: vector<64xf16>, %b: vector<[64]xf16>) -> ui64 {
  // expected-error @+1 {{operand #1 must be fixed-length vector}}
  %0 = "aievec.aie4.fcmp"(%a, %b) {predicate = #aievec.aie4_cmp_predicate<lt>} : (vector<64xf16>, vector<[64]xf16>) -> ui64
  return %0 : ui64
}

// -----

func.func @fcmp_rank_two_operand0(%a: vector<2x16xbf16>, %b: vector<32xbf16>) -> ui32 {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "aievec.aie4.fcmp"(%a, %b) {predicate = #aievec.aie4_cmp_predicate<lt>} : (vector<2x16xbf16>, vector<32xbf16>) -> ui32
  return %0 : ui32
}

// -----

func.func @fcmp_rank_two_operand1(%a: vector<64xf16>, %b: vector<2x32xf16>) -> ui64 {
  // expected-error @+1 {{operand #1 must be fixed-length vector}}
  %0 = "aievec.aie4.fcmp"(%a, %b) {predicate = #aievec.aie4_cmp_predicate<lt>} : (vector<64xf16>, vector<2x32xf16>) -> ui64
  return %0 : ui64
}

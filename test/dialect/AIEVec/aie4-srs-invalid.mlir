// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics

func.func @shift_negative(%a : vector<64xi32>) -> vector<64xi8> {
  // expected-error @+1 {{attribute 'shift' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0 whose maximum value is 59}}
  %0 = aievec.aie4.srs %a {shift = -1 : i32, isSigned = true} : vector<64xi32> -> vector<64xi8>
  return %0 : vector<64xi8>
}

// -----

// Encodings 60..63 are left shifts and are not exposed.
func.func @shift_too_large(%a : vector<32xi32>) -> vector<32xi16> {
  // expected-error @+1 {{attribute 'shift' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0 whose maximum value is 59}}
  %0 = aievec.aie4.srs %a {shift = 60 : i32, isSigned = true} : vector<32xi32> -> vector<32xi16>
  return %0 : vector<32xi16>
}

// -----

func.func @v64_to_v32(%a : vector<64xi32>) -> vector<32xi16> {
  // expected-error @+1 {{result must have as many lanes as the source: vector<64xi32> -> vector<64xi8> or vector<32xi32> -> vector<32xi16>, got 'vector<64xi32>' -> 'vector<32xi16>'}}
  %0 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = true} : vector<64xi32> -> vector<32xi16>
  return %0 : vector<32xi16>
}

// -----

func.func @v32_to_v64(%a : vector<32xi32>) -> vector<64xi8> {
  // expected-error @+1 {{result must have as many lanes as the source}}
  %0 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = true} : vector<32xi32> -> vector<64xi8>
  return %0 : vector<64xi8>
}

// -----

func.func @v16_source(%a : vector<16xi32>) -> vector<16xi8> {
  // expected-error @+1 {{operand #0 must be fixed-length vector of 32-bit signless integer values of length 64 of ranks 1 or fixed-length vector of 32-bit signless integer values of length 32 of ranks 1, but got 'vector<16xi32>'}}
  %0 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = true} : vector<16xi32> -> vector<16xi8>
  return %0 : vector<16xi8>
}

// -----

func.func @v64_to_i16(%a : vector<64xi32>) -> vector<64xi16> {
  // expected-error @+1 {{result #0 must be fixed-length vector of 8-bit signless integer values of length 64 of ranks 1 or fixed-length vector of 16-bit signless integer values of length 32 of ranks 1, but got 'vector<64xi16>'}}
  %0 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = true} : vector<64xi32> -> vector<64xi16>
  return %0 : vector<64xi16>
}

// -----

func.func @scalable_source(%a : vector<[64]xi32>) -> vector<64xi8> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = true} : vector<[64]xi32> -> vector<64xi8>
  return %0 : vector<64xi8>
}

// -----

func.func @rank2_source(%a : vector<2x32xi32>) -> vector<64xi8> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = true} : vector<2x32xi32> -> vector<64xi8>
  return %0 : vector<64xi8>
}

// -----

// isSigned is required: an explicit-mode op does not assume signedness.
func.func @missing_is_signed(%a : vector<64xi32>) -> vector<64xi8> {
  // expected-error @+1 {{requires attribute 'isSigned'}}
  %0 = aievec.aie4.srs %a {shift = 4 : i32} : vector<64xi32> -> vector<64xi8>
  return %0 : vector<64xi8>
}

// -----

// Stochastic rounding (16) is not exposed.
func.func @stochastic(%a : vector<64xi32>) -> vector<64xi8> {
  // expected-error @+2 {{failed to parse AIEVec_AIE4RoundingModeAttr parameter 'value' which is to be a `::xilinx::aievec::AIE4RoundingMode`}}
  // expected-error @+1 {{expected ::xilinx::aievec::AIE4RoundingMode to be one of: floor, ceil, sym_floor, sym_ceil, neg_inf, pos_inf, sym_zero, sym_inf, conv_even, conv_odd}}
  %0 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = true, rounding = #aievec.aie4_rnd<stochastic>} : vector<64xi32> -> vector<64xi8>
  return %0 : vector<64xi8>
}

// -----

// Encoding 0 is spelled 'truncate', not 'none'.
func.func @sat_none(%a : vector<64xi32>) -> vector<64xi8> {
  // expected-error @+2 {{failed to parse AIEVec_AIE4SaturationModeAttr parameter 'value' which is to be a `::xilinx::aievec::AIE4SaturationMode`}}
  // expected-error @+1 {{expected ::xilinx::aievec::AIE4SaturationMode to be one of: truncate, saturate, symmetric}}
  %0 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = true, saturation = #aievec.aie4_sat<none>} : vector<64xi32> -> vector<64xi8>
  return %0 : vector<64xi8>
}

// -----

func.func @sat_two(%a : vector<64xi32>) -> vector<64xi8> {
  // expected-error @+2 {{failed to parse AIEVec_AIE4SaturationModeAttr parameter 'value' which is to be a `::xilinx::aievec::AIE4SaturationMode`}}
  // expected-error @+1 {{expected valid keyword or string}}
  %0 = aievec.aie4.srs %a {shift = 4 : i32, isSigned = true, saturation = #aievec.aie4_sat<2>} : vector<64xi32> -> vector<64xi8>
  return %0 : vector<64xi8>
}

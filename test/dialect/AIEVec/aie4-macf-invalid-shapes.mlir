// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Each operand and result must be a fixed, rank-one vector of 32 lanes.

func.func @f32_mac_scalable_operand0(%a0: vector<[32]xf32>, %a1: vector<32xf32>, %a2: vector<32xf32>) -> vector<32xf32> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_f32"(%a0, %a1, %a2) : (vector<[32]xf32>, vector<32xf32>, vector<32xf32>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @f32_mac_scalable_operand1(%a0: vector<32xf32>, %a1: vector<[32]xf32>, %a2: vector<32xf32>) -> vector<32xf32> {
  // expected-error @+1 {{operand #1 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_f32"(%a0, %a1, %a2) : (vector<32xf32>, vector<[32]xf32>, vector<32xf32>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @f32_mac_scalable_operand2(%a0: vector<32xf32>, %a1: vector<32xf32>, %a2: vector<[32]xf32>) -> vector<32xf32> {
  // expected-error @+1 {{operand #2 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_f32"(%a0, %a1, %a2) : (vector<32xf32>, vector<32xf32>, vector<[32]xf32>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @f32_mac_scalable_result(%a0: vector<32xf32>, %a1: vector<32xf32>, %a2: vector<32xf32>) -> vector<[32]xf32> {
  // expected-error @+1 {{result #0 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_f32"(%a0, %a1, %a2) : (vector<32xf32>, vector<32xf32>, vector<32xf32>) -> vector<[32]xf32>
  return %0 : vector<[32]xf32>
}

// -----

func.func @f32_mac_rank_two_operand0(%a0: vector<2x16xf32>, %a1: vector<32xf32>, %a2: vector<32xf32>) -> vector<32xf32> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_f32"(%a0, %a1, %a2) : (vector<2x16xf32>, vector<32xf32>, vector<32xf32>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @f32_mac_rank_two_operand1(%a0: vector<32xf32>, %a1: vector<2x16xf32>, %a2: vector<32xf32>) -> vector<32xf32> {
  // expected-error @+1 {{operand #1 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_f32"(%a0, %a1, %a2) : (vector<32xf32>, vector<2x16xf32>, vector<32xf32>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @f32_mac_rank_two_operand2(%a0: vector<32xf32>, %a1: vector<32xf32>, %a2: vector<2x16xf32>) -> vector<32xf32> {
  // expected-error @+1 {{operand #2 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_f32"(%a0, %a1, %a2) : (vector<32xf32>, vector<32xf32>, vector<2x16xf32>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @f32_mac_rank_two_result(%a0: vector<32xf32>, %a1: vector<32xf32>, %a2: vector<32xf32>) -> vector<2x16xf32> {
  // expected-error @+1 {{result #0 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_f32"(%a0, %a1, %a2) : (vector<32xf32>, vector<32xf32>, vector<32xf32>) -> vector<2x16xf32>
  return %0 : vector<2x16xf32>
}

// -----

func.func @bf16_mul_scalable_operand0(%a0: vector<[32]xbf16>, %a1: vector<32xbf16>) -> vector<32xf32> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "aievec.aie4.mul_ewise_bf16_f32"(%a0, %a1) : (vector<[32]xbf16>, vector<32xbf16>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @bf16_mul_scalable_operand1(%a0: vector<32xbf16>, %a1: vector<[32]xbf16>) -> vector<32xf32> {
  // expected-error @+1 {{operand #1 must be fixed-length vector}}
  %0 = "aievec.aie4.mul_ewise_bf16_f32"(%a0, %a1) : (vector<32xbf16>, vector<[32]xbf16>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @bf16_mul_scalable_result(%a0: vector<32xbf16>, %a1: vector<32xbf16>) -> vector<[32]xf32> {
  // expected-error @+1 {{result #0 must be fixed-length vector}}
  %0 = "aievec.aie4.mul_ewise_bf16_f32"(%a0, %a1) : (vector<32xbf16>, vector<32xbf16>) -> vector<[32]xf32>
  return %0 : vector<[32]xf32>
}

// -----

func.func @bf16_mul_rank_two_operand0(%a0: vector<2x16xbf16>, %a1: vector<32xbf16>) -> vector<32xf32> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "aievec.aie4.mul_ewise_bf16_f32"(%a0, %a1) : (vector<2x16xbf16>, vector<32xbf16>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @bf16_mul_rank_two_operand1(%a0: vector<32xbf16>, %a1: vector<2x16xbf16>) -> vector<32xf32> {
  // expected-error @+1 {{operand #1 must be fixed-length vector}}
  %0 = "aievec.aie4.mul_ewise_bf16_f32"(%a0, %a1) : (vector<32xbf16>, vector<2x16xbf16>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @bf16_mul_rank_two_result(%a0: vector<32xbf16>, %a1: vector<32xbf16>) -> vector<2x16xf32> {
  // expected-error @+1 {{result #0 must be fixed-length vector}}
  %0 = "aievec.aie4.mul_ewise_bf16_f32"(%a0, %a1) : (vector<32xbf16>, vector<32xbf16>) -> vector<2x16xf32>
  return %0 : vector<2x16xf32>
}

// -----

func.func @bf16_mac_scalable_operand0(%a0: vector<[32]xbf16>, %a1: vector<32xbf16>, %a2: vector<32xf32>) -> vector<32xf32> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_bf16_f32"(%a0, %a1, %a2) : (vector<[32]xbf16>, vector<32xbf16>, vector<32xf32>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @bf16_mac_scalable_operand1(%a0: vector<32xbf16>, %a1: vector<[32]xbf16>, %a2: vector<32xf32>) -> vector<32xf32> {
  // expected-error @+1 {{operand #1 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_bf16_f32"(%a0, %a1, %a2) : (vector<32xbf16>, vector<[32]xbf16>, vector<32xf32>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @bf16_mac_scalable_operand2(%a0: vector<32xbf16>, %a1: vector<32xbf16>, %a2: vector<[32]xf32>) -> vector<32xf32> {
  // expected-error @+1 {{operand #2 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_bf16_f32"(%a0, %a1, %a2) : (vector<32xbf16>, vector<32xbf16>, vector<[32]xf32>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @bf16_mac_scalable_result(%a0: vector<32xbf16>, %a1: vector<32xbf16>, %a2: vector<32xf32>) -> vector<[32]xf32> {
  // expected-error @+1 {{result #0 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_bf16_f32"(%a0, %a1, %a2) : (vector<32xbf16>, vector<32xbf16>, vector<32xf32>) -> vector<[32]xf32>
  return %0 : vector<[32]xf32>
}

// -----

func.func @bf16_mac_rank_two_operand0(%a0: vector<2x16xbf16>, %a1: vector<32xbf16>, %a2: vector<32xf32>) -> vector<32xf32> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_bf16_f32"(%a0, %a1, %a2) : (vector<2x16xbf16>, vector<32xbf16>, vector<32xf32>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @bf16_mac_rank_two_operand1(%a0: vector<32xbf16>, %a1: vector<2x16xbf16>, %a2: vector<32xf32>) -> vector<32xf32> {
  // expected-error @+1 {{operand #1 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_bf16_f32"(%a0, %a1, %a2) : (vector<32xbf16>, vector<2x16xbf16>, vector<32xf32>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @bf16_mac_rank_two_operand2(%a0: vector<32xbf16>, %a1: vector<32xbf16>, %a2: vector<2x16xf32>) -> vector<32xf32> {
  // expected-error @+1 {{operand #2 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_bf16_f32"(%a0, %a1, %a2) : (vector<32xbf16>, vector<32xbf16>, vector<2x16xf32>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @bf16_mac_rank_two_result(%a0: vector<32xbf16>, %a1: vector<32xbf16>, %a2: vector<32xf32>) -> vector<2x16xf32> {
  // expected-error @+1 {{result #0 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_bf16_f32"(%a0, %a1, %a2) : (vector<32xbf16>, vector<32xbf16>, vector<32xf32>) -> vector<2x16xf32>
  return %0 : vector<2x16xf32>
}

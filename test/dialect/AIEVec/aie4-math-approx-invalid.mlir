// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// VEXP2/VTANH and their XLLVM wrappers take exactly vector<32xf32>.

func.func @biased_exp2_bf16_source(%x: vector<32xbf16>) -> vector<32xbf16> {
  // expected-error @+1 {{operand #0 must be fixed-length vector of 32-bit float values of length 32}}
  %0 = "aievec.aie4.biased_exp2"(%x) : (vector<32xbf16>) -> vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

func.func @biased_exp2_f16_result(%x: vector<32xf32>) -> vector<32xf16> {
  // expected-error @+1 {{result #0 must be}}
  %0 = "aievec.aie4.biased_exp2"(%x) : (vector<32xf32>) -> vector<32xf16>
  return %0 : vector<32xf16>
}

// -----

func.func @biased_exp2_64_lanes(%x: vector<64xf32>) -> vector<64xf32> {
  // expected-error @+1 {{operand #0 must be}}
  %0 = "aievec.aie4.biased_exp2"(%x) : (vector<64xf32>) -> vector<64xf32>
  return %0 : vector<64xf32>
}

// -----

func.func @tanh_f32_result(%x: vector<32xf32>) -> vector<32xf32> {
  // expected-error @+1 {{result #0 must be fixed-length vector of bfloat16 type values of length 32}}
  %0 = "aievec.aie4.tanh"(%x) : (vector<32xf32>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @tanh_scalable(%x: vector<[32]xf32>) -> vector<[32]xbf16> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "aievec.aie4.tanh"(%x) : (vector<[32]xf32>) -> vector<[32]xbf16>
  return %0 : vector<[32]xbf16>
}

// -----

func.func @xllvm_exp2_f32_result(%x: vector<32xf32>) -> vector<32xf32> {
  // expected-error @+1 {{failed to verify that 'res' is fixed-length vector of bfloat16 type values of length 32}}
  %0 = "xllvm.intr.aie4.exp2"(%x) : (vector<32xf32>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @xllvm_exp2_bf20_bf16_result(%x: vector<32xf32>) -> vector<32xbf16> {
  // expected-error @+1 {{failed to verify that 'res' is fixed-length vector of 32-bit float values of length 32}}
  %0 = "xllvm.intr.aie4.exp2.bf20"(%x) : (vector<32xf32>) -> vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

func.func @xllvm_tanh_bf16_source(%x: vector<32xbf16>) -> vector<32xbf16> {
  // expected-error @+1 {{operand #0 must be}}
  %0 = "xllvm.intr.aie4.tanh"(%x) : (vector<32xbf16>) -> vector<32xbf16>
  return %0 : vector<32xbf16>
}

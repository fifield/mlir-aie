// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie4b" --convert-vector-to-aievec="aie-target=aie4b" -split-input-file -verify-diagnostics

// Strict exp2/tanh never take the approximate native path. AIE4 has no strict
// lowering (Peano aborts on G_FEXP2/G_FTANH), so they are diagnosed here.

func.func @exp2_strict(%x: vector<32xf32>) -> vector<32xf32> {
  // expected-warning @+2 {{AIE4: math.exp2 requires fastmath<afn>; AIE4 has only the approximate VEXP2 lowering and no strict lowering}}
  // expected-error @+1 {{failed to legalize operation 'math.exp2' that was explicitly marked illegal}}
  %0 = math.exp2 %x : vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @tanh_strict(%x: vector<32xf32>) -> vector<32xf32> {
  // expected-warning @+2 {{AIE4: math.tanh requires fastmath<afn>; AIE4 has only the approximate VTANH lowering and no strict lowering}}
  // expected-error @+1 {{failed to legalize operation 'math.tanh' that was explicitly marked illegal}}
  %0 = math.tanh %x : vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @exp2_strict_bf16(%x: vector<32xbf16>) -> vector<32xbf16> {
  // expected-warning @+2 {{requires fastmath<afn>}}
  // expected-error @+1 {{failed to legalize operation 'math.exp2'}}
  %0 = math.exp2 %x : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

// Other fast-math flags do not grant approximation.
func.func @exp2_no_afn(%x: vector<32xf32>) -> vector<32xf32> {
  // expected-warning @+2 {{requires fastmath<afn>}}
  // expected-error @+1 {{failed to legalize operation 'math.exp2'}}
  %0 = math.exp2 %x fastmath<reassoc,nnan,ninf,nsz,arcp,contract> : vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @tanh_no_afn(%x: vector<32xbf16>) -> vector<32xbf16> {
  // expected-warning @+2 {{requires fastmath<afn>}}
  // expected-error @+1 {{failed to legalize operation 'math.tanh'}}
  %0 = math.tanh %x fastmath<nnan,ninf> : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

// Unsupported shapes and element types.

func.func @exp2_scalar(%x: f32) -> f32 {
  // expected-warning @+2 {{AIE4: afn math.exp2 is only supported on fixed 1-D vectors}}
  // expected-error @+1 {{failed to legalize operation 'math.exp2'}}
  %0 = math.exp2 %x fastmath<afn> : f32
  return %0 : f32
}

// -----

func.func @tanh_16_lanes(%x: vector<16xf32>) -> vector<16xf32> {
  // expected-warning @+2 {{AIE4: afn math.tanh requires a multiple of 32 lanes, not 16}}
  // expected-error @+1 {{failed to legalize operation 'math.tanh'}}
  %0 = math.tanh %x fastmath<afn> : vector<16xf32>
  return %0 : vector<16xf32>
}

// -----

func.func @exp2_48_lanes(%x: vector<48xbf16>) -> vector<48xbf16> {
  // expected-warning @+2 {{AIE4: afn math.exp2 requires a multiple of 32 lanes, not 48}}
  // expected-error @+1 {{failed to legalize operation 'math.exp2'}}
  %0 = math.exp2 %x fastmath<afn> : vector<48xbf16>
  return %0 : vector<48xbf16>
}

// -----

func.func @exp2_scalable(%x: vector<[32]xf32>) -> vector<[32]xf32> {
  // expected-warning @+2 {{only supported on fixed 1-D vectors}}
  // expected-error @+1 {{failed to legalize operation 'math.exp2'}}
  %0 = math.exp2 %x fastmath<afn> : vector<[32]xf32>
  return %0 : vector<[32]xf32>
}

// -----

func.func @tanh_rank2(%x: vector<2x32xf32>) -> vector<2x32xf32> {
  // expected-warning @+2 {{only supported on fixed 1-D vectors}}
  // expected-error @+1 {{failed to legalize operation 'math.tanh'}}
  %0 = math.tanh %x fastmath<afn> : vector<2x32xf32>
  return %0 : vector<2x32xf32>
}

// -----

func.func @exp2_f16(%x: vector<32xf16>) -> vector<32xf16> {
  // expected-warning @+2 {{AIE4: afn math.exp2 only supports f32/bf16 elements, not 'f16'}}
  // expected-error @+1 {{failed to legalize operation 'math.exp2'}}
  %0 = math.exp2 %x fastmath<afn> : vector<32xf16>
  return %0 : vector<32xf16>
}

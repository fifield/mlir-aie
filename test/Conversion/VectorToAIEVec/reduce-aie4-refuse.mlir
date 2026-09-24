// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie4b" --convert-vector-to-aievec="aie-target=aie4b" -split-input-file -verify-diagnostics

// Unsupported non-add reductions report why native lowering is unavailable.

// Native floating min/max propagate NaNs, unlike minnumf/maxnumf.
func.func @reduce_minnumf_refused(%a: vector<32xf32>) -> f32 {
  // expected-warning @+2 {{minnumf/maxnumf cannot be lowered}}
  // expected-error @+1 {{failed to legalize operation 'vector.reduction' that was explicitly marked illegal}}
  %0 = vector.reduction <minnumf>, %a : vector<32xf32> into f32
  return %0 : f32
}

// -----

func.func @reduce_maxnumf_refused(%a: vector<32xf32>) -> f32 {
  // expected-warning @+2 {{minnumf/maxnumf cannot be lowered}}
  // expected-error @+1 {{failed to legalize operation 'vector.reduction' that was explicitly marked illegal}}
  %0 = vector.reduction <maxnumf>, %a : vector<32xf32> into f32
  return %0 : f32
}

// -----

// Native integer min/max are signed only.
func.func @reduce_minui_refused(%a: vector<32xi32>) -> i32 {
  // expected-warning @+2 {{no unsigned vector reduce instruction}}
  // expected-error @+1 {{failed to legalize operation 'vector.reduction' that was explicitly marked illegal}}
  %0 = vector.reduction <minui>, %a : vector<32xi32> into i32
  return %0 : i32
}

// -----

func.func @reduce_maxui_refused(%a: vector<32xi32>) -> i32 {
  // expected-warning @+2 {{no unsigned vector reduce instruction}}
  // expected-error @+1 {{failed to legalize operation 'vector.reduction' that was explicitly marked illegal}}
  %0 = vector.reduction <maxui>, %a : vector<32xi32> into i32
  return %0 : i32
}

// -----

// No native reduction exists for these combining kinds.
func.func @reduce_mul_refused(%a: vector<32xi32>) -> i32 {
  // expected-warning @+2 {{no vector reduce instruction for this combining kind}}
  // expected-error @+1 {{failed to legalize operation 'vector.reduction' that was explicitly marked illegal}}
  %0 = vector.reduction <mul>, %a : vector<32xi32> into i32
  return %0 : i32
}

// -----

func.func @reduce_and_refused(%a: vector<32xi32>) -> i32 {
  // expected-warning @+2 {{no vector reduce instruction for this combining kind}}
  // expected-error @+1 {{failed to legalize operation 'vector.reduction' that was explicitly marked illegal}}
  %0 = vector.reduction <and>, %a : vector<32xi32> into i32
  return %0 : i32
}

// -----

func.func @reduce_or_refused(%a: vector<32xi32>) -> i32 {
  // expected-warning @+2 {{no vector reduce instruction for this combining kind}}
  // expected-error @+1 {{failed to legalize operation 'vector.reduction' that was explicitly marked illegal}}
  %0 = vector.reduction <or>, %a : vector<32xi32> into i32
  return %0 : i32
}

// -----

func.func @reduce_xor_refused(%a: vector<32xi32>) -> i32 {
  // expected-warning @+2 {{no vector reduce instruction for this combining kind}}
  // expected-error @+1 {{failed to legalize operation 'vector.reduction' that was explicitly marked illegal}}
  %0 = vector.reduction <xor>, %a : vector<32xi32> into i32
  return %0 : i32
}

// -----

// A scalable 32-lane minimum cannot use a fixed-width native instruction.
func.func @reduce_scalable_minsi_refused(%a: vector<[32]xi32>) -> i32 {
  // expected-warning @+2 {{only fixed 1-D vector.reduction is supported}}
  // expected-error @+1 {{failed to legalize operation 'vector.reduction' that was explicitly marked illegal}}
  %0 = vector.reduction <minsi>, %a : vector<[32]xi32> into i32
  return %0 : i32
}

// -----

// 512-bit signed min has no matching native instruction.
func.func @reduce_512bit_refused(%a: vector<16xi32>) -> i32 {
  // expected-warning @+2 {{no reduce instruction exists at 512-bit width}}
  // expected-error @+1 {{failed to legalize operation 'vector.reduction' that was explicitly marked illegal}}
  %0 = vector.reduction <minsi>, %a : vector<16xi32> into i32
  return %0 : i32
}

// -----

func.func @reduce_i16_refused(%a: vector<64xi16>) -> i16 {
  // expected-warning @+2 {{only supports i32/f32 elements}}
  // expected-error @+1 {{failed to legalize operation 'vector.reduction' that was explicitly marked illegal}}
  %0 = vector.reduction <minsi>, %a : vector<64xi16> into i16
  return %0 : i16
}

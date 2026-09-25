// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Both operands and the intrinsic result must be fixed vectors of 16 i32 lanes.

// -----

func.func @shuffle_rank_two_lhs(%a: vector<4x4xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // expected-error@+1 {{operand #0 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.vshuffle"(%a, %b) <{mode = 66 : i32}> : (vector<4x4xi32>, vector<16xi32>) -> vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

func.func @shuffle_rank_two_rhs(%a: vector<16xi32>, %b: vector<4x4xi32>) -> vector<16xi32> {
  // expected-error@+1 {{operand #1 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.vshuffle"(%a, %b) <{mode = 66 : i32}> : (vector<16xi32>, vector<4x4xi32>) -> vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

func.func @shuffle_rank_two_result(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<4x4xi32> {
  // expected-error@+1 {{result #0 must be LLVM dialect-compatible type}}
  %0 = "xllvm.intr.aie4.vshuffle"(%a, %b) <{mode = 66 : i32}> : (vector<16xi32>, vector<16xi32>) -> vector<4x4xi32>
  return %0 : vector<4x4xi32>
}

// -----

func.func @shuffle_scalable_lhs(%a: vector<[16]xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // expected-error@+1 {{operand #0 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.vshuffle"(%a, %b) <{mode = 66 : i32}> : (vector<[16]xi32>, vector<16xi32>) -> vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

func.func @shuffle_scalable_rhs(%a: vector<16xi32>, %b: vector<[16]xi32>) -> vector<16xi32> {
  // expected-error@+1 {{operand #1 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.vshuffle"(%a, %b) <{mode = 66 : i32}> : (vector<16xi32>, vector<[16]xi32>) -> vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

func.func @shuffle_scalable_result(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<[16]xi32> {
  // expected-error@+1 {{'res' is fixed-length vector}}
  %0 = "xllvm.intr.aie4.vshuffle"(%a, %b) <{mode = 66 : i32}> : (vector<16xi32>, vector<16xi32>) -> vector<[16]xi32>
  return %0 : vector<[16]xi32>
}

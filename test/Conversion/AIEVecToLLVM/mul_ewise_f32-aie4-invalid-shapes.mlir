// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Direct XLLVM input must obey the intrinsic's fixed 32xf32 signature too.
func.func @scalable_lhs(%a: vector<[32]xf32>, %b: vector<32xf32>) -> vector<32xf32> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.I1024.I1024.ACC1024.fp.mul.ewise.conf"(%a, %b) {zeroAcc, amode = 1 : i32, bmode = 3 : i32} : (vector<[32]xf32>, vector<32xf32>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @rank_two_rhs(%a: vector<32xf32>, %b: vector<4x8xf32>) -> vector<32xf32> {
  // expected-error @+1 {{operand #1 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.I1024.I1024.ACC1024.fp.mul.ewise.conf"(%a, %b) {zeroAcc, amode = 1 : i32, bmode = 3 : i32} : (vector<32xf32>, vector<4x8xf32>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

func.func @scalable_result(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<[32]xf32> {
  // expected-error @+1 {{'res' is fixed-length vector}}
  %0 = "xllvm.intr.aie4.I1024.I1024.ACC1024.fp.mul.ewise.conf"(%a, %b) {zeroAcc, amode = 1 : i32, bmode = 3 : i32} : (vector<32xf32>, vector<32xf32>) -> vector<[32]xf32>
  return %0 : vector<[32]xf32>
}

// -----

func.func @rank_two_result(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<4x8xf32> {
  // The base LLVM result constraint rejects multidimensional vectors first.
  // expected-error @+1 {{result #0 must be LLVM dialect-compatible type}}
  %0 = "xllvm.intr.aie4.I1024.I1024.ACC1024.fp.mul.ewise.conf"(%a, %b) {zeroAcc, amode = 1 : i32, bmode = 3 : i32} : (vector<32xf32>, vector<32xf32>) -> vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

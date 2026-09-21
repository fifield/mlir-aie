// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Native MAC/SRS operations require fixed, rank-one vectors.

func.func @mac_scalable_lhs(%a: vector<[32]xi16>, %b: vector<32xi16>, %acc: vector<32xi32>) -> vector<32xi32> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.I512.I512.ACC1024.mac.ewise.conf"(%a, %b, %acc) {amode = 0 : i32, bmode = 1 : i32, lhsSign = 1 : i32, rhsSign = 1 : i32} : (vector<[32]xi16>, vector<32xi16>, vector<32xi32>) -> vector<32xi32>
  return %0 : vector<32xi32>
}

// -----

func.func @mac_rank_two_rhs(%a: vector<32xi16>, %b: vector<4x8xi16>, %acc: vector<32xi32>) -> vector<32xi32> {
  // expected-error @+1 {{operand #1 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.I512.I512.ACC1024.mac.ewise.conf"(%a, %b, %acc) {amode = 0 : i32, bmode = 1 : i32, lhsSign = 1 : i32, rhsSign = 1 : i32} : (vector<32xi16>, vector<4x8xi16>, vector<32xi32>) -> vector<32xi32>
  return %0 : vector<32xi32>
}

// -----

func.func @mac_scalable_acc(%a: vector<32xi16>, %b: vector<32xi16>, %acc: vector<[32]xi32>) -> vector<32xi32> {
  // expected-error @+1 {{operand #2 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.I512.I512.ACC1024.mac.ewise.conf"(%a, %b, %acc) {amode = 0 : i32, bmode = 1 : i32, lhsSign = 1 : i32, rhsSign = 1 : i32} : (vector<32xi16>, vector<32xi16>, vector<[32]xi32>) -> vector<32xi32>
  return %0 : vector<32xi32>
}

// -----

func.func @mac_scalable_result(%a: vector<32xi16>, %b: vector<32xi16>, %acc: vector<32xi32>) -> vector<[32]xi32> {
  // expected-error @+1 {{'res' is fixed-length vector}}
  %0 = "xllvm.intr.aie4.I512.I512.ACC1024.mac.ewise.conf"(%a, %b, %acc) {amode = 0 : i32, bmode = 1 : i32, lhsSign = 1 : i32, rhsSign = 1 : i32} : (vector<32xi16>, vector<32xi16>, vector<32xi32>) -> vector<[32]xi32>
  return %0 : vector<[32]xi32>
}

// -----

func.func @srs_scalable_source(%a: vector<[32]xi32>) -> vector<32xi16> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.I512.v32.acc32.srs"(%a) {shift = 4 : i32, sign = 1 : i32} : (vector<[32]xi32>) -> vector<32xi16>
  return %0 : vector<32xi16>
}

// -----

func.func @srs_rank_two_source(%a: vector<4x8xi32>) -> vector<32xi16> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.I512.v32.acc32.srs"(%a) {shift = 4 : i32, sign = 1 : i32} : (vector<4x8xi32>) -> vector<32xi16>
  return %0 : vector<32xi16>
}

// -----

func.func @srs_scalable_result(%a: vector<32xi32>) -> vector<[32]xi16> {
  // expected-error @+1 {{'res' is fixed-length vector}}
  %0 = "xllvm.intr.aie4.I512.v32.acc32.srs"(%a) {shift = 4 : i32, sign = 1 : i32} : (vector<32xi32>) -> vector<[32]xi16>
  return %0 : vector<[32]xi16>
}

// -----

func.func @srs_rank_two_result(%a: vector<32xi32>) -> vector<4x8xi16> {
  // expected-error @+1 {{result #0 must be LLVM dialect-compatible type}}
  %0 = "xllvm.intr.aie4.I512.v32.acc32.srs"(%a) {shift = 4 : i32, sign = 1 : i32} : (vector<32xi32>) -> vector<4x8xi16>
  return %0 : vector<4x8xi16>
}

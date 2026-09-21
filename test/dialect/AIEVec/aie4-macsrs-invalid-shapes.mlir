// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Native MAC/SRS operations require fixed, rank-one vectors.

func.func @mac_scalable_lhs(%a: vector<[32]xi16>, %b: vector<32xi16>, %acc: vector<32xi32>) -> vector<32xi32> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_i16_i32"(%a, %b, %acc) : (vector<[32]xi16>, vector<32xi16>, vector<32xi32>) -> vector<32xi32>
  return %0 : vector<32xi32>
}

// -----

func.func @mac_rank_two_rhs(%a: vector<32xi16>, %b: vector<4x8xi16>, %acc: vector<32xi32>) -> vector<32xi32> {
  // expected-error @+1 {{operand #1 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_i16_i32"(%a, %b, %acc) : (vector<32xi16>, vector<4x8xi16>, vector<32xi32>) -> vector<32xi32>
  return %0 : vector<32xi32>
}

// -----

func.func @mac_scalable_acc(%a: vector<32xi16>, %b: vector<32xi16>, %acc: vector<[32]xi32>) -> vector<32xi32> {
  // expected-error @+1 {{operand #2 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_i16_i32"(%a, %b, %acc) : (vector<32xi16>, vector<32xi16>, vector<[32]xi32>) -> vector<32xi32>
  return %0 : vector<32xi32>
}

// -----

func.func @mac_scalable_result(%a: vector<32xi16>, %b: vector<32xi16>, %acc: vector<32xi32>) -> vector<[32]xi32> {
  // expected-error @+1 {{result #0 must be fixed-length vector}}
  %0 = "aievec.aie4.mac_ewise_i16_i32"(%a, %b, %acc) : (vector<32xi16>, vector<32xi16>, vector<32xi32>) -> vector<[32]xi32>
  return %0 : vector<[32]xi32>
}

// -----

func.func @srs_scalable_source(%a: vector<[32]xi32>) -> vector<32xi16> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "aievec.aie4.srs_i32_i16"(%a) {shift = 4 : i32} : (vector<[32]xi32>) -> vector<32xi16>
  return %0 : vector<32xi16>
}

// -----

func.func @srs_rank_two_source(%a: vector<4x8xi32>) -> vector<32xi16> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "aievec.aie4.srs_i32_i16"(%a) {shift = 4 : i32} : (vector<4x8xi32>) -> vector<32xi16>
  return %0 : vector<32xi16>
}

// -----

func.func @srs_scalable_result(%a: vector<32xi32>) -> vector<[32]xi16> {
  // expected-error @+1 {{result #0 must be fixed-length vector}}
  %0 = "aievec.aie4.srs_i32_i16"(%a) {shift = 4 : i32} : (vector<32xi32>) -> vector<[32]xi16>
  return %0 : vector<[32]xi16>
}

// -----

func.func @srs_rank_two_result(%a: vector<32xi32>) -> vector<4x8xi16> {
  // expected-error @+1 {{result #0 must be fixed-length vector}}
  %0 = "aievec.aie4.srs_i32_i16"(%a) {shift = 4 : i32} : (vector<32xi32>) -> vector<4x8xi16>
  return %0 : vector<4x8xi16>
}

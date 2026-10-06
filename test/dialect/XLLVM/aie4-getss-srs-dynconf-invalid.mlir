// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics

func.func @get_ss_kind_out_of_range(%p: i32) -> vector<32xi32> {
  // expected-error @+1 {{attribute 'kind' failed to satisfy constraint: AIE4 1024-bit stream read kind}}
  %0 = "xllvm.intr.aie4.get.ss"(%p) {kind = 3 : i32} : (i32) -> vector<32xi32>
  return %0 : vector<32xi32>
}

// -----

func.func @get_ss_missing_kind(%p: i32) -> vector<32xi32> {
  // expected-error @+1 {{requires attribute 'kind'}}
  %0 = "xllvm.intr.aie4.get.ss"(%p) : (i32) -> vector<32xi32>
  return %0 : vector<32xi32>
}

// -----

// The result must match aie.get_stream's <32 x i32> declaration.
func.func @get_ss_byte_result(%p: i32) -> vector<128xi8> {
  // expected-error @+1 {{failed to verify that 'res' is fixed-length vector of 32-bit signless integer values of length 32}}
  %0 = "xllvm.intr.aie4.get.ss"(%p) {kind = 0 : i32} : (i32) -> vector<128xi8>
  return %0 : vector<128xi8>
}

// -----

func.func @get_ss_index_port(%p: index) -> vector<32xi32> {
  // expected-error @+1 {{operand #0 must be 32-bit signless integer}}
  %0 = "xllvm.intr.aie4.get.ss"(%p) {kind = 0 : i32} : (index) -> vector<32xi32>
  return %0 : vector<32xi32>
}

// -----

// Reject shifts outside the supported right-shift range.
func.func @srs_v64_shift_too_large(%acc: vector<64xi32>) -> vector<64xi8> {
  // expected-error @+1 {{attribute 'shift' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0 whose maximum value is 59}}
  %0 = "xllvm.intr.aie4.I512.v64.acc32.srs"(%acc) {shift = 60 : i32, sign = 1 : i32} : (vector<64xi32>) -> vector<64xi8>
  return %0 : vector<64xi8>
}

// -----

func.func @srs_v64_negative_shift(%acc: vector<64xi32>) -> vector<64xi8> {
  // expected-error @+1 {{attribute 'shift' failed to satisfy constraint}}
  %0 = "xllvm.intr.aie4.I512.v64.acc32.srs"(%acc) {shift = -1 : i32, sign = 1 : i32} : (vector<64xi32>) -> vector<64xi8>
  return %0 : vector<64xi8>
}

// -----

func.func @srs_v64_sign_out_of_range(%acc: vector<64xi32>) -> vector<64xi8> {
  // expected-error @+1 {{attribute 'sign' failed to satisfy constraint}}
  %0 = "xllvm.intr.aie4.I512.v64.acc32.srs"(%acc) {shift = 12 : i32, sign = 2 : i32} : (vector<64xi32>) -> vector<64xi8>
  return %0 : vector<64xi8>
}

// -----

func.func @srs_v64_i16_result(%acc: vector<64xi32>) -> vector<64xi16> {
  // expected-error @+1 {{failed to verify that 'res' is fixed-length vector of 8-bit signless integer values of length 64}}
  %0 = "xllvm.intr.aie4.I512.v64.acc32.srs"(%acc) {shift = 12 : i32, sign = 1 : i32} : (vector<64xi32>) -> vector<64xi16>
  return %0 : vector<64xi16>
}

// -----

func.func @dyn_conf_i64_conf(%acc: vector<64xi32>, %a: vector<64xi8>, %c: i64) -> vector<64xi32> {
  // expected-error @+1 {{operand #2 must be 32-bit signless integer}}
  %0 = "xllvm.intr.aie4.I512.ACC2048.mmac.dyn_conf"(%acc, %a, %c) : (vector<64xi32>, vector<64xi8>, i64) -> vector<64xi32>
  return %0 : vector<64xi32>
}

// -----

// Integer formats only; the f32 accumulator belongs to bf16 and fp8.
func.func @dyn_conf_float_acc(%acc: vector<64xf32>, %a: vector<64xi8>, %c: i32) -> vector<64xf32> {
  // expected-error @+1 {{operand #0 must be fixed-length vector of 32-bit signless integer values of length 64}}
  %0 = "xllvm.intr.aie4.I512.ACC2048.mmac.dyn_conf"(%acc, %a, %c) : (vector<64xf32>, vector<64xi8>, i32) -> vector<64xf32>
  return %0 : vector<64xf32>
}

// -----

func.func @dyn_conf_static_attr(%acc: vector<64xi32>, %a: vector<64xi8>) -> vector<64xi32> {
  // expected-error @+1 {{expected 3 operands, but found 2}}
  %0 = "xllvm.intr.aie4.I512.ACC2048.mmac.dyn_conf"(%acc, %a) {bmode = 1 : i32} : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
  return %0 : vector<64xi32>
}

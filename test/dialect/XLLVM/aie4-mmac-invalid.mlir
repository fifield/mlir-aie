// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Configuration enums reject values that cannot be encoded in the conf word.

func.func @int_bmode_out_of_range(%acc: vector<64xi32>, %a: vector<64xi8>) -> vector<64xi32> {
  // expected-error @+1 {{attribute 'bmode' failed to satisfy constraint: AIE4 dense integer MMAC precision (conf bit 3)}}
  %0 = "xllvm.intr.aie4.I512.ACC2048.mmac"(%acc, %a) {bmode = 2 : i32, xSign = 1 : i32, zSign = 1 : i32} : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
  return %0 : vector<64xi32>
}

// -----

func.func @int_sign_out_of_range(%acc: vector<64xi32>, %a: vector<64xi8>) -> vector<64xi32> {
  // expected-error @+1 {{attribute 'zSign' failed to satisfy constraint}}
  %0 = "xllvm.intr.aie4.I512.ACC2048.mmac"(%acc, %a) {bmode = 1 : i32, xSign = 1 : i32, zSign = 2 : i32} : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
  return %0 : vector<64xi32>
}

// -----

func.func @int_missing_sign(%acc: vector<64xi32>, %a: vector<64xi8>) -> vector<64xi32> {
  // expected-error @+1 {{requires attribute 'xSign'}}
  %0 = "xllvm.intr.aie4.I512.ACC2048.mmac"(%acc, %a) {bmode = 1 : i32, zSign = 1 : i32} : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
  return %0 : vector<64xi32>
}

// -----

func.func @fp8_emode_undefined(%acc: vector<64xf32>, %a: vector<64xi8>) -> vector<64xf32> {
  // expected-error @+1 {{attribute 'emode' failed to satisfy constraint: AIE4 fp8 MMAC X-operand NaN processing (conf bits 11:10)}}
  %0 = "xllvm.intr.aie4.fp8.I512.ACC2048.mmac"(%acc, %a) {emode = 3 : i32} : (vector<64xf32>, vector<64xi8>) -> vector<64xf32>
  return %0 : vector<64xf32>
}

// -----

func.func @staging_mode_out_of_range(%b: vector<128xi8>, %i: i32) {
  // expected-error @+1 {{attribute 'mode' failed to satisfy constraint: AIE4 integer staging mode written to crIntMMode}}
  "xllvm.intr.aie4.insert.staging.int.I1024"(%b, %i) {mode = 4 : i32} : (vector<128xi8>, i32) -> ()
  return
}

// -----

// The dense forms take one 512-bit activation row; 1024-bit rows belong to
// the sparse intrinsics.
func.func @int_sparse_width_row(%acc: vector<64xi32>, %a: vector<128xi8>) -> vector<64xi32> {
  // expected-error @+1 {{operand #1 must be fixed-length vector of 8-bit signless integer values of length 64}}
  %0 = "xllvm.intr.aie4.I512.ACC2048.mmac"(%acc, %a) {bmode = 1 : i32, xSign = 1 : i32, zSign = 1 : i32} : (vector<64xi32>, vector<128xi8>) -> vector<64xi32>
  return %0 : vector<64xi32>
}

// -----

func.func @bf16_wrong_acc(%acc: vector<32xf32>, %a: vector<32xbf16>) -> vector<64xf32> {
  // expected-error @+1 {{operand #0 must be fixed-length vector of 32-bit float values of length 64}}
  %0 = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%acc, %a) : (vector<32xf32>, vector<32xbf16>) -> vector<64xf32>
  return %0 : vector<64xf32>
}

// -----

func.func @bf16_wrong_result(%acc: vector<64xf32>, %a: vector<32xbf16>) -> vector<64xi32> {
  // expected-error @+1 {{'res' is fixed-length vector of 32-bit float values of length 64}}
  %0 = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%acc, %a) : (vector<64xf32>, vector<32xbf16>) -> vector<64xi32>
  return %0 : vector<64xi32>
}

// -----

func.func @staging_half_chunk(%b: vector<64xi8>, %i: i32) {
  // expected-error @+1 {{operand #0 must be fixed-length vector of 8-bit signless integer values of length 128}}
  "xllvm.intr.aie4.insert.staging.int.I1024"(%b, %i) {mode = 3 : i32} : (vector<64xi8>, i32) -> ()
  return
}

// -----

func.func @commit_with_result() -> i32 {
  // expected-error @+1 {{requires zero results}}
  %0 = "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> i32
  return %0 : i32
}

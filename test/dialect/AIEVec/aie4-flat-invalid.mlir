// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Verifier errors of the flat matrix-state ops.

//===----------------------------------------------------------------------===//
// stage: chunk type per format
//===----------------------------------------------------------------------===//

func.func @stage_bf16_bytes(%c: vector<128xi8>, %s: i32) {
  // expected-error @+1 {{chunk must be 'vector<64xbf16>' for format bf16, got 'vector<128xi8>'}}
  aievec.aie4.stage bf16 %c, %s : vector<128xi8>, i32
  return
}

// -----

func.func @stage_int8_short(%c: vector<64xi8>, %s: i32) {
  // expected-error @+1 {{chunk must be 'vector<128xi8>' for format int8, got 'vector<64xi8>'}}
  aievec.aie4.stage int8 %c, %s {rhs_signed = true} : vector<64xi8>, i32
  return
}

// -----

func.func @stage_int4_wrong(%c: vector<128xi4>, %s: i32) {
  // expected-error @+1 {{chunk must be 'vector<256xi4>' or 'vector<128xi8>' for format int4, got 'vector<128xi4>'}}
  aievec.aie4.stage int4 %c, %s {rhs_signed = true} : vector<128xi4>, i32
  return
}

// -----

func.func @stage_fp8_wrong(%c: vector<64xbf16>, %s: i32) {
  // expected-error @+1 {{chunk must be 'vector<128xf8E4M3FN>' or 'vector<128xi8>' for format fp8, got 'vector<64xbf16>'}}
  aievec.aie4.stage fp8 %c, %s : vector<64xbf16>, i32
  return
}

// -----

// The byte image is accepted for int4 and fp8 only.
func.func @stage_int8_logical_int4(%c: vector<256xi4>, %s: i32) {
  // expected-error @+1 {{chunk must be 'vector<128xi8>' for format int8, got 'vector<256xi4>'}}
  aievec.aie4.stage int8 %c, %s {rhs_signed = true} : vector<256xi4>, i32
  return
}

//===----------------------------------------------------------------------===//
// stage and stage_stream: signedness, slot and port
//===----------------------------------------------------------------------===//

// -----

func.func @stage_missing_rhs_signed(%c: vector<128xi8>, %s: i32) {
  // expected-error @+1 {{integer format requires 'rhs_signed'}}
  aievec.aie4.stage int8 %c, %s : vector<128xi8>, i32
  return
}

// -----

func.func @stage_float_rhs_signed(%c: vector<64xbf16>, %s: i32) {
  // expected-error @+1 {{'rhs_signed' applies only to integer formats}}
  aievec.aie4.stage bf16 %c, %s {rhs_signed = true} : vector<64xbf16>, i32
  return
}

// -----

func.func @stage_slot_negative(%c: vector<128xi8>) {
  %s = arith.constant -1 : i32
  // expected-error @+1 {{slot -1 is out of range [0, 32)}}
  aievec.aie4.stage int8 %c, %s {rhs_signed = true} : vector<128xi8>, i32
  return
}

// -----

func.func @stage_slot_32(%c: vector<64xbf16>) {
  %s = arith.constant 32 : i32
  // expected-error @+1 {{slot 32 is out of range [0, 32)}}
  aievec.aie4.stage bf16 %c, %s : vector<64xbf16>, i32
  return
}

// -----

func.func @stage_stream_slot_32(%p: i32) {
  %s = arith.constant 32 : i32
  // expected-error @+1 {{slot 32 is out of range [0, 32)}}
  aievec.aie4.stage_stream bf16 %p, %s : i32, i32
  return
}

// -----

func.func @stage_stream_port_2(%s: i32) {
  %p = arith.constant 2 : i32
  // expected-error @+1 {{port 2 is not an AIE4 core input stream (0 or 1)}}
  aievec.aie4.stage_stream int8 %p, %s {rhs_signed = true} : i32, i32
  return
}

// -----

func.func @stage_stream_missing_rhs_signed(%p: i32, %s: i32) {
  // expected-error @+1 {{integer format requires 'rhs_signed'}}
  aievec.aie4.stage_stream int4 %p, %s : i32, i32
  return
}

// -----

func.func @stage_stream_float_rhs_signed(%p: i32, %s: i32) {
  // expected-error @+1 {{'rhs_signed' applies only to integer formats}}
  aievec.aie4.stage_stream fp8 %p, %s {rhs_signed = false} : i32, i32
  return
}

//===----------------------------------------------------------------------===//
// commit
//===----------------------------------------------------------------------===//

// -----

func.func @commit_missing_rhs_signed() {
  // expected-error @+1 {{integer format requires 'rhs_signed'}}
  aievec.aie4.commit int4
  return
}

// -----

func.func @commit_float_rhs_signed() {
  // expected-error @+1 {{'rhs_signed' applies only to integer formats}}
  aievec.aie4.commit bf16 {rhs_signed = true}
  return
}

//===----------------------------------------------------------------------===//
// mmac_active: shared row-MMAC checks
//===----------------------------------------------------------------------===//

// -----

func.func @mmac_active_bad_lhs(%a: vector<64xbf16>, %c: vector<64xf32>) {
  // expected-error @+1 {{lhs must be 'vector<32xbf16>' for format bf16, got 'vector<64xbf16>'}}
  %r = aievec.aie4.mmac_active bf16 %a, %c : vector<64xbf16>, vector<64xf32> -> vector<64xf32>
  return
}

// -----

func.func @mmac_active_bad_acc(%a: vector<64xi8>, %c: vector<64xf32>) {
  // expected-error @+1 {{acc must be 'vector<64xi32>' for format int8, got 'vector<64xf32>'}}
  %r = aievec.aie4.mmac_active int8 %a, %c {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xf32> -> vector<64xi32>
  return
}

// -----

func.func @mmac_active_bad_result(%a: vector<64xf8E4M3FN>) {
  // expected-error @+1 {{result must be 'vector<64xf32>' for format fp8, got 'vector<64xi32>'}}
  %r = aievec.aie4.mmac_active fp8 %a : vector<64xf8E4M3FN> -> vector<64xi32>
  return
}

// -----

func.func @mmac_active_missing_lhs_signed(%a: vector<128xi4>, %c: vector<64xi32>) {
  // expected-error @+1 {{integer format requires 'lhs_signed'}}
  %r = aievec.aie4.mmac_active int4 %a, %c {rhs_signed = true} : vector<128xi4>, vector<64xi32> -> vector<64xi32>
  return
}

// -----

func.func @mmac_active_missing_rhs_signed(%a: vector<64xi8>, %c: vector<64xi32>) {
  // expected-error @+1 {{integer format requires 'rhs_signed'}}
  %r = aievec.aie4.mmac_active int8 %a, %c {lhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
  return
}

// -----

func.func @mmac_active_float_signed(%a: vector<32xbf16>, %c: vector<64xf32>) {
  // expected-error @+1 {{'lhs_signed' applies only to integer formats}}
  %r = aievec.aie4.mmac_active bf16 %a, %c {lhs_signed = true} : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  return
}

//===----------------------------------------------------------------------===//
// mmac_active: zero_if
//===----------------------------------------------------------------------===//

// -----

func.func @mmac_active_zero_if_bf16(%a: vector<32xbf16>, %c: vector<64xf32>, %z: i1) {
  // expected-error @+1 {{'zero_if' applies only to integer formats (int8, int4); float formats peel the first K tile}}
  %r = aievec.aie4.mmac_active bf16 %a, %c zero_if %z : vector<32xbf16>, vector<64xf32> -> vector<64xf32>
  return
}

// -----

func.func @mmac_active_zero_if_fp8(%a: vector<64xf8E4M3FN>, %c: vector<64xf32>, %z: i1) {
  // expected-error @+1 {{'zero_if' applies only to integer formats}}
  %r = aievec.aie4.mmac_active fp8 %a, %c zero_if %z : vector<64xf8E4M3FN>, vector<64xf32> -> vector<64xf32>
  return
}

// -----

func.func @mmac_active_zero_if_no_acc(%a: vector<64xi8>, %z: i1) {
  // expected-error @+1 {{'zero_if' requires 'acc'; without 'acc' the accumulator is always zero}}
  %r = aievec.aie4.mmac_active int8 %a zero_if %z {lhs_signed = true, rhs_signed = true} : vector<64xi8> -> vector<64xi32>
  return
}

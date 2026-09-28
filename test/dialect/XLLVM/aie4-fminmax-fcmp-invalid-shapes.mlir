// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -verify-diagnostics

func.func @bf16_512_scalable_operand0(%a: vector<[32]xbf16>, %b: vector<32xbf16>) -> !llvm.struct<(vector<32xbf16>, i32)> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.vaddbfloat16.512"(%a, %b) {opsel = 20 : i32} : (vector<[32]xbf16>, vector<32xbf16>) -> !llvm.struct<(vector<32xbf16>, i32)>
  return %0 : !llvm.struct<(vector<32xbf16>, i32)>
}

// -----

func.func @bf16_512_rank_two_operand1(%a: vector<32xbf16>, %b: vector<2x16xbf16>) -> !llvm.struct<(vector<32xbf16>, i32)> {
  // expected-error @+1 {{operand #1 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.vaddbfloat16.512"(%a, %b) {opsel = 20 : i32} : (vector<32xbf16>, vector<2x16xbf16>) -> !llvm.struct<(vector<32xbf16>, i32)>
  return %0 : !llvm.struct<(vector<32xbf16>, i32)>
}

// -----

func.func @bf16_512_scalable_result(%a: vector<32xbf16>, %b: vector<32xbf16>) -> !llvm.struct<(vector<[32]xbf16>, i32)> {
  // expected-error @+1 {{'res' is an LLVM struct}}
  %0 = "xllvm.intr.aie4.vaddbfloat16.512"(%a, %b) {opsel = 20 : i32} : (vector<32xbf16>, vector<32xbf16>) -> !llvm.struct<(vector<[32]xbf16>, i32)>
  return %0 : !llvm.struct<(vector<[32]xbf16>, i32)>
}

// -----

func.func @bf16_1024_scalable_operand0(%a: vector<[64]xbf16>, %b: vector<64xbf16>) -> !llvm.struct<(vector<64xbf16>, vector<2xi32>)> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.vaddbfloat16.1024"(%a, %b) {opsel = 24 : i32} : (vector<[64]xbf16>, vector<64xbf16>) -> !llvm.struct<(vector<64xbf16>, vector<2xi32>)>
  return %0 : !llvm.struct<(vector<64xbf16>, vector<2xi32>)>
}

// -----

func.func @bf16_1024_rank_two_operand1(%a: vector<64xbf16>, %b: vector<2x32xbf16>) -> !llvm.struct<(vector<64xbf16>, vector<2xi32>)> {
  // expected-error @+1 {{operand #1 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.vaddbfloat16.1024"(%a, %b) {opsel = 24 : i32} : (vector<64xbf16>, vector<2x32xbf16>) -> !llvm.struct<(vector<64xbf16>, vector<2xi32>)>
  return %0 : !llvm.struct<(vector<64xbf16>, vector<2xi32>)>
}

// -----

func.func @bf16_1024_scalable_result(%a: vector<64xbf16>, %b: vector<64xbf16>) -> !llvm.struct<(vector<[64]xbf16>, vector<2xi32>)> {
  // expected-error @+1 {{'res' is an LLVM struct}}
  %0 = "xllvm.intr.aie4.vaddbfloat16.1024"(%a, %b) {opsel = 24 : i32} : (vector<64xbf16>, vector<64xbf16>) -> !llvm.struct<(vector<[64]xbf16>, vector<2xi32>)>
  return %0 : !llvm.struct<(vector<[64]xbf16>, vector<2xi32>)>
}

// -----

func.func @bf16_1024_scalable_mask(%a: vector<64xbf16>, %b: vector<64xbf16>) -> !llvm.struct<(vector<64xbf16>, vector<[2]xi32>)> {
  // expected-error @+1 {{'res' is an LLVM struct}}
  %0 = "xllvm.intr.aie4.vaddbfloat16.1024"(%a, %b) {opsel = 24 : i32} : (vector<64xbf16>, vector<64xbf16>) -> !llvm.struct<(vector<64xbf16>, vector<[2]xi32>)>
  return %0 : !llvm.struct<(vector<64xbf16>, vector<[2]xi32>)>
}

// -----

func.func @fp16_512_scalable_operand0(%a: vector<[32]xf16>, %b: vector<32xf16>) -> !llvm.struct<(vector<32xf16>, i32)> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.vaddfloat16.512"(%a, %b) {opsel = 21 : i32} : (vector<[32]xf16>, vector<32xf16>) -> !llvm.struct<(vector<32xf16>, i32)>
  return %0 : !llvm.struct<(vector<32xf16>, i32)>
}

// -----

func.func @fp16_512_rank_two_operand1(%a: vector<32xf16>, %b: vector<2x16xf16>) -> !llvm.struct<(vector<32xf16>, i32)> {
  // expected-error @+1 {{operand #1 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.vaddfloat16.512"(%a, %b) {opsel = 21 : i32} : (vector<32xf16>, vector<2x16xf16>) -> !llvm.struct<(vector<32xf16>, i32)>
  return %0 : !llvm.struct<(vector<32xf16>, i32)>
}

// -----

func.func @fp16_512_scalable_result(%a: vector<32xf16>, %b: vector<32xf16>) -> !llvm.struct<(vector<[32]xf16>, i32)> {
  // expected-error @+1 {{'res' is an LLVM struct}}
  %0 = "xllvm.intr.aie4.vaddfloat16.512"(%a, %b) {opsel = 21 : i32} : (vector<32xf16>, vector<32xf16>) -> !llvm.struct<(vector<[32]xf16>, i32)>
  return %0 : !llvm.struct<(vector<[32]xf16>, i32)>
}

// -----

func.func @fp16_1024_scalable_operand0(%a: vector<[64]xf16>, %b: vector<64xf16>) -> !llvm.struct<(vector<64xf16>, vector<2xi32>)> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.vaddfloat16.1024"(%a, %b) {opsel = 25 : i32} : (vector<[64]xf16>, vector<64xf16>) -> !llvm.struct<(vector<64xf16>, vector<2xi32>)>
  return %0 : !llvm.struct<(vector<64xf16>, vector<2xi32>)>
}

// -----

func.func @fp16_1024_rank_two_operand1(%a: vector<64xf16>, %b: vector<2x32xf16>) -> !llvm.struct<(vector<64xf16>, vector<2xi32>)> {
  // expected-error @+1 {{operand #1 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.vaddfloat16.1024"(%a, %b) {opsel = 25 : i32} : (vector<64xf16>, vector<2x32xf16>) -> !llvm.struct<(vector<64xf16>, vector<2xi32>)>
  return %0 : !llvm.struct<(vector<64xf16>, vector<2xi32>)>
}

// -----

func.func @fp16_1024_scalable_result(%a: vector<64xf16>, %b: vector<64xf16>) -> !llvm.struct<(vector<[64]xf16>, vector<2xi32>)> {
  // expected-error @+1 {{'res' is an LLVM struct}}
  %0 = "xllvm.intr.aie4.vaddfloat16.1024"(%a, %b) {opsel = 25 : i32} : (vector<64xf16>, vector<64xf16>) -> !llvm.struct<(vector<[64]xf16>, vector<2xi32>)>
  return %0 : !llvm.struct<(vector<[64]xf16>, vector<2xi32>)>
}

// -----

func.func @fp16_1024_scalable_mask(%a: vector<64xf16>, %b: vector<64xf16>) -> !llvm.struct<(vector<64xf16>, vector<[2]xi32>)> {
  // expected-error @+1 {{'res' is an LLVM struct}}
  %0 = "xllvm.intr.aie4.vaddfloat16.1024"(%a, %b) {opsel = 25 : i32} : (vector<64xf16>, vector<64xf16>) -> !llvm.struct<(vector<64xf16>, vector<[2]xi32>)>
  return %0 : !llvm.struct<(vector<64xf16>, vector<[2]xi32>)>
}

// -----

func.func @fp32_512_scalable_operand0(%a: vector<[16]xf32>, %b: vector<16xf32>) -> !llvm.struct<(vector<16xf32>, i32)> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.vaddfloat.512"(%a, %b) {opsel = 36 : i32} : (vector<[16]xf32>, vector<16xf32>) -> !llvm.struct<(vector<16xf32>, i32)>
  return %0 : !llvm.struct<(vector<16xf32>, i32)>
}

// -----

func.func @fp32_512_rank_two_operand1(%a: vector<16xf32>, %b: vector<2x8xf32>) -> !llvm.struct<(vector<16xf32>, i32)> {
  // expected-error @+1 {{operand #1 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.vaddfloat.512"(%a, %b) {opsel = 36 : i32} : (vector<16xf32>, vector<2x8xf32>) -> !llvm.struct<(vector<16xf32>, i32)>
  return %0 : !llvm.struct<(vector<16xf32>, i32)>
}

// -----

func.func @fp32_512_scalable_result(%a: vector<16xf32>, %b: vector<16xf32>) -> !llvm.struct<(vector<[16]xf32>, i32)> {
  // expected-error @+1 {{'res' is an LLVM struct}}
  %0 = "xllvm.intr.aie4.vaddfloat.512"(%a, %b) {opsel = 36 : i32} : (vector<16xf32>, vector<16xf32>) -> !llvm.struct<(vector<[16]xf32>, i32)>
  return %0 : !llvm.struct<(vector<[16]xf32>, i32)>
}

// -----

func.func @fp32_1024_scalable_operand0(%a: vector<[32]xf32>, %b: vector<32xf32>) -> !llvm.struct<(vector<32xf32>, i32)> {
  // expected-error @+1 {{operand #0 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.vaddfloat.1024"(%a, %b) {opsel = 37 : i32} : (vector<[32]xf32>, vector<32xf32>) -> !llvm.struct<(vector<32xf32>, i32)>
  return %0 : !llvm.struct<(vector<32xf32>, i32)>
}

// -----

func.func @fp32_1024_rank_two_operand1(%a: vector<32xf32>, %b: vector<2x16xf32>) -> !llvm.struct<(vector<32xf32>, i32)> {
  // expected-error @+1 {{operand #1 must be fixed-length vector}}
  %0 = "xllvm.intr.aie4.vaddfloat.1024"(%a, %b) {opsel = 37 : i32} : (vector<32xf32>, vector<2x16xf32>) -> !llvm.struct<(vector<32xf32>, i32)>
  return %0 : !llvm.struct<(vector<32xf32>, i32)>
}

// -----

func.func @fp32_1024_scalable_result(%a: vector<32xf32>, %b: vector<32xf32>) -> !llvm.struct<(vector<[32]xf32>, i32)> {
  // expected-error @+1 {{'res' is an LLVM struct}}
  %0 = "xllvm.intr.aie4.vaddfloat.1024"(%a, %b) {opsel = 37 : i32} : (vector<32xf32>, vector<32xf32>) -> !llvm.struct<(vector<[32]xf32>, i32)>
  return %0 : !llvm.struct<(vector<[32]xf32>, i32)>
}

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" | FileCheck %s

// Check intrinsic selection, result fields, and conversion between scalar
// masks and the vector<2xi32> carrier used for 64-lane shapes.

// -- MINMAX --

// CHECK-LABEL: @minmax_i8x64_min_signed
func.func @minmax_i8x64_min_signed(%a: vector<64xi8>, %b: vector<64xi8>) -> vector<64xi8> {
  // CHECK: %[[R:.*]] = "xllvm.intr.aie4.vadd8.512"
  // CHECK-SAME: opsel = 13 : i32
  // CHECK-SAME: sign = 1 : i32
  // CHECK-SAME: -> !llvm.struct<(vector<64xi8>, vector<2xi32>)>
  // CHECK: llvm.extractvalue %[[R]][0]
  %0 = aievec.aie4.minmax %a, %b [kind = <min>, signed = true] : vector<64xi8>
  return %0 : vector<64xi8>
}

// CHECK-LABEL: @minmax_i32x16_max_unsigned
func.func @minmax_i32x16_max_unsigned(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // CHECK: %[[R:.*]] = "xllvm.intr.aie4.vadd32.512"
  // CHECK-SAME: opsel = 14 : i32
  // CHECK-SAME: sign = 0 : i32
  // CHECK-SAME: -> !llvm.struct<(vector<16xi32>, i32)>
  // CHECK: llvm.extractvalue %[[R]][0]
  %0 = aievec.aie4.minmax %a, %b [kind = <max>, signed = false] : vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

// -- CMP --

// CHECK-LABEL: @cmp_i16x32_lt_signed
func.func @cmp_i16x32_lt_signed(%a: vector<32xi16>, %b: vector<32xi16>) -> ui32 {
  // CHECK: %[[R:.*]] = "xllvm.intr.aie4.vadd16.512"
  // CHECK-SAME: opsel = 15 : i32
  // CHECK-SAME: sign = 1 : i32
  // CHECK-SAME: -> !llvm.struct<(vector<32xi16>, i32)>
  // CHECK: %[[MASK:.*]] = llvm.extractvalue %[[R]][1]
  // Native i32 mask carrier: no bitcast needed before the result is returned.
  // CHECK-NOT: llvm.bitcast
  // CHECK: builtin.unrealized_conversion_cast %[[MASK]]
  %0 = aievec.aie4.cmp %a, %b [pred = <lt>, signed = true] : vector<32xi16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: @cmp_i8x64_ge_unsigned
func.func @cmp_i8x64_ge_unsigned(%a: vector<64xi8>, %b: vector<64xi8>) -> ui64 {
  // CHECK: %[[R:.*]] = "xllvm.intr.aie4.vadd8.512"
  // CHECK-SAME: opsel = 17 : i32
  // CHECK-SAME: sign = 0 : i32
  // CHECK-SAME: -> !llvm.struct<(vector<64xi8>, vector<2xi32>)>
  // CHECK: %[[MASK:.*]] = llvm.extractvalue %[[R]][1]
  // 64-bit register-pair mask carrier: must be bitcast <2 x i32> -> i64.
  // CHECK: llvm.bitcast %[[MASK]] : vector<2xi32> to i64
  %0 = aievec.aie4.cmp %a, %b [pred = <ge>, signed = false] : vector<64xi8>, ui64
  return %0 : ui64
}

// -----

// -- SEL --

// CHECK-LABEL: @sel_i32x32
func.func @sel_i32x32(%a: vector<32xi32>, %b: vector<32xi32>, %sel: ui32) -> vector<32xi32> {
  // Native i32 mask carrier: no bitcast needed before the sel operand is used.
  // CHECK: %[[SEL:.*]] = builtin.unrealized_conversion_cast
  // CHECK-NOT: llvm.bitcast
  // CHECK: "xllvm.intr.aie4.vadd32.mask.1024"
  // CHECK-SAME: %[[SEL]]
  // CHECK-SAME: -> vector<32xi32>
  %0 = aievec.aie4.sel %a, %b, %sel : vector<32xi32>, ui32, vector<32xi32>
  return %0 : vector<32xi32>
}

// CHECK-LABEL: @sel_i16x64
func.func @sel_i16x64(%a: vector<64xi16>, %b: vector<64xi16>, %sel: ui64) -> vector<64xi16> {
  // 64-bit register-pair mask carrier: must be bitcast i64 -> <2 x i32> before
  // being passed to the intrinsic.
  // CHECK: %[[SEL64:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[SEL:.*]] = llvm.bitcast %[[SEL64]] : i64 to vector<2xi32>
  // CHECK: "xllvm.intr.aie4.vadd16.mask.1024"
  // CHECK-SAME: %[[SEL]]
  // CHECK-SAME: -> vector<64xi16>
  %0 = aievec.aie4.sel %a, %b, %sel : vector<64xi16>, ui64, vector<64xi16>
  return %0 : vector<64xi16>
}

// -----

// -- FLOAT MINMAX/CMP --
//
// Check XLLVM selectors and result types for all six supported float shapes.
// FP32 uses Ge=48 and Lt=49, reversing the selector order of BF16 and FP16.

// CHECK-LABEL: @fminmax_bf16x32_min
func.func @fminmax_bf16x32_min(%a: vector<32xbf16>, %b: vector<32xbf16>) -> vector<32xbf16> {
  // CHECK: %[[R:.*]] = "xllvm.intr.aie4.vaddbfloat16.512"
  // CHECK-SAME: opsel = 20 : i32
  // CHECK-SAME: -> !llvm.struct<(vector<32xbf16>, i32)>
  // CHECK: llvm.extractvalue %[[R]][0]
  %0 = aievec.aie4.fminmax %a, %b [kind = <min>] : vector<32xbf16>
  return %0 : vector<32xbf16>
}

// CHECK-LABEL: @fminmax_bf16x64_max
func.func @fminmax_bf16x64_max(%a: vector<64xbf16>, %b: vector<64xbf16>) -> vector<64xbf16> {
  // CHECK: %[[R:.*]] = "xllvm.intr.aie4.vaddbfloat16.1024"
  // CHECK-SAME: opsel = 24 : i32
  // CHECK-SAME: -> !llvm.struct<(vector<64xbf16>, vector<2xi32>)>
  // CHECK: llvm.extractvalue %[[R]][0]
  %0 = aievec.aie4.fminmax %a, %b [kind = <max>] : vector<64xbf16>
  return %0 : vector<64xbf16>
}

// CHECK-LABEL: @fminmax_fp16x32_min
func.func @fminmax_fp16x32_min(%a: vector<32xf16>, %b: vector<32xf16>) -> vector<32xf16> {
  // CHECK: %[[R:.*]] = "xllvm.intr.aie4.vaddfloat16.512"
  // CHECK-SAME: opsel = 21 : i32
  // CHECK-SAME: -> !llvm.struct<(vector<32xf16>, i32)>
  // CHECK: llvm.extractvalue %[[R]][0]
  %0 = aievec.aie4.fminmax %a, %b [kind = <min>] : vector<32xf16>
  return %0 : vector<32xf16>
}

// CHECK-LABEL: @fminmax_fp16x64_max
func.func @fminmax_fp16x64_max(%a: vector<64xf16>, %b: vector<64xf16>) -> vector<64xf16> {
  // CHECK: %[[R:.*]] = "xllvm.intr.aie4.vaddfloat16.1024"
  // CHECK-SAME: opsel = 25 : i32
  // CHECK-SAME: -> !llvm.struct<(vector<64xf16>, vector<2xi32>)>
  // CHECK: llvm.extractvalue %[[R]][0]
  %0 = aievec.aie4.fminmax %a, %b [kind = <max>] : vector<64xf16>
  return %0 : vector<64xf16>
}

// CHECK-LABEL: @fminmax_fp32x16_min
func.func @fminmax_fp32x16_min(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[R:.*]] = "xllvm.intr.aie4.vaddfloat.512"
  // CHECK-SAME: opsel = 36 : i32
  // CHECK-SAME: -> !llvm.struct<(vector<16xf32>, i32)>
  // CHECK: llvm.extractvalue %[[R]][0]
  %0 = aievec.aie4.fminmax %a, %b [kind = <min>] : vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: @fminmax_fp32x32_max
func.func @fminmax_fp32x32_max(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<32xf32> {
  // CHECK: %[[R:.*]] = "xllvm.intr.aie4.vaddfloat.1024"
  // CHECK-SAME: opsel = 37 : i32
  // CHECK-SAME: -> !llvm.struct<(vector<32xf32>, i32)>
  // CHECK: llvm.extractvalue %[[R]][0]
  %0 = aievec.aie4.fminmax %a, %b [kind = <max>] : vector<32xf32>
  return %0 : vector<32xf32>
}

// -----

// -- FLOAT CMP --

// CHECK-LABEL: @fcmp_bf16x32_lt
func.func @fcmp_bf16x32_lt(%a: vector<32xbf16>, %b: vector<32xbf16>) -> ui32 {
  // CHECK: %[[R:.*]] = "xllvm.intr.aie4.vaddbfloat16.512"
  // CHECK-SAME: opsel = 40 : i32
  // CHECK-SAME: -> !llvm.struct<(vector<32xbf16>, i32)>
  // CHECK: %[[MASK:.*]] = llvm.extractvalue %[[R]][1]
  // CHECK: builtin.unrealized_conversion_cast %[[MASK]]
  %0 = aievec.aie4.fcmp %a, %b [pred = <lt>] : vector<32xbf16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: @fcmp_bf16x64_ge
func.func @fcmp_bf16x64_ge(%a: vector<64xbf16>, %b: vector<64xbf16>) -> ui64 {
  // CHECK: %[[R:.*]] = "xllvm.intr.aie4.vaddbfloat16.1024"
  // CHECK-SAME: opsel = 44 : i32
  // CHECK-SAME: -> !llvm.struct<(vector<64xbf16>, vector<2xi32>)>
  // CHECK: %[[MASK:.*]] = llvm.extractvalue %[[R]][1]
  // 64-bit register-pair mask carrier: must be bitcast <2 x i32> -> i64.
  // CHECK: llvm.bitcast %[[MASK]] : vector<2xi32> to i64
  %0 = aievec.aie4.fcmp %a, %b [pred = <ge>] : vector<64xbf16>, ui64
  return %0 : ui64
}

// CHECK-LABEL: @fcmp_fp16x32_lt
func.func @fcmp_fp16x32_lt(%a: vector<32xf16>, %b: vector<32xf16>) -> ui32 {
  // CHECK: %[[R:.*]] = "xllvm.intr.aie4.vaddfloat16.512"
  // CHECK-SAME: opsel = 41 : i32
  // CHECK: %[[MASK:.*]] = llvm.extractvalue %[[R]][1]
  // CHECK: builtin.unrealized_conversion_cast %[[MASK]]
  %0 = aievec.aie4.fcmp %a, %b [pred = <lt>] : vector<32xf16>, ui32
  return %0 : ui32
}

// CHECK-LABEL: @fcmp_fp16x64_ge
func.func @fcmp_fp16x64_ge(%a: vector<64xf16>, %b: vector<64xf16>) -> ui64 {
  // CHECK: %[[R:.*]] = "xllvm.intr.aie4.vaddfloat16.1024"
  // CHECK-SAME: opsel = 45 : i32
  // CHECK: %[[MASK:.*]] = llvm.extractvalue %[[R]][1]
  // CHECK: llvm.bitcast %[[MASK]] : vector<2xi32> to i64
  %0 = aievec.aie4.fcmp %a, %b [pred = <ge>] : vector<64xf16>, ui64
  return %0 : ui64
}

// Check both FP32 selectors explicitly to catch an accidental lt/ge swap.
// CHECK-LABEL: @fcmp_fp32x16_ge
func.func @fcmp_fp32x16_ge(%a: vector<16xf32>, %b: vector<16xf32>) -> ui32 {
  // CHECK: %[[R:.*]] = "xllvm.intr.aie4.vaddfloat.512"
  // CHECK-SAME: opsel = 48 : i32
  // CHECK: %[[MASK:.*]] = llvm.extractvalue %[[R]][1]
  // CHECK: builtin.unrealized_conversion_cast %[[MASK]]
  %0 = aievec.aie4.fcmp %a, %b [pred = <ge>] : vector<16xf32>, ui32
  return %0 : ui32
}

// CHECK-LABEL: @fcmp_fp32x32_lt
func.func @fcmp_fp32x32_lt(%a: vector<32xf32>, %b: vector<32xf32>) -> ui32 {
  // CHECK: %[[R:.*]] = "xllvm.intr.aie4.vaddfloat.1024"
  // CHECK-SAME: opsel = 49 : i32
  // CHECK: %[[MASK:.*]] = llvm.extractvalue %[[R]][1]
  // CHECK: builtin.unrealized_conversion_cast %[[MASK]]
  %0 = aievec.aie4.fcmp %a, %b [pred = <lt>] : vector<32xf32>, ui32
  return %0 : ui32
}

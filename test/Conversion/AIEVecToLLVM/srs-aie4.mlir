// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// aievec.aie4.srs lowers to the v64 / v32 XLLVM SRS. Explicit modes lower
// where the function's aie4.entry_modes (or aie4-assume-modes) guarantees
// them.
//
// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" | FileCheck %s
// With aie4-assume-modes, the same cases lower too (the attribute, where
// present, stays authoritative); srs-aie4-assume-modes.mlir covers a function
// without the attribute.
// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b aie4-assume-modes=rnd:9,sat:1" | FileCheck %s

// Mode-less v64: signed.
// CHECK-LABEL: func.func @modeless_v64
// CHECK: "xllvm.intr.aie4.I512.v64.acc32.srs"(%{{.*}}) <{shift = 12 : i32, sign = 1 : i32}> : (vector<64xi32>) -> vector<64xi8>
func.func @modeless_v64(%acc : vector<64xi32>) -> vector<64xi8> {
  %0 = aievec.aie4.srs %acc {shift = 12 : i32, isSigned = true} : vector<64xi32> -> vector<64xi8>
  return %0 : vector<64xi8>
}

// -----

// Mode-less v32: unsigned.
// CHECK-LABEL: func.func @modeless_v32_unsigned
// CHECK: "xllvm.intr.aie4.I512.v32.acc32.srs"(%{{.*}}) <{shift = 3 : i32, sign = 0 : i32}> : (vector<32xi32>) -> vector<32xi16>
func.func @modeless_v32_unsigned(%acc : vector<32xi32>) -> vector<32xi16> {
  %0 = aievec.aie4.srs %acc {shift = 3 : i32, isSigned = false} : vector<32xi32> -> vector<32xi16>
  return %0 : vector<32xi16>
}

// -----

// Both modes guaranteed by the entry modes.
// CHECK-LABEL: func.func @explicit_both
// CHECK: "xllvm.intr.aie4.I512.v64.acc32.srs"(%{{.*}}) <{shift = 12 : i32, sign = 1 : i32}>
func.func @explicit_both(%acc : vector<64xi32>) -> vector<64xi8>
    attributes {aie4.entry_modes = {rnd = 9 : i32, sat = 1 : i32}} {
  %0 = aievec.aie4.srs %acc {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  return %0 : vector<64xi8>
}

// -----

// Rounding only, against an attribute that fixes only crRnd.
// CHECK-LABEL: func.func @explicit_rnd_only
// CHECK: "xllvm.intr.aie4.I512.v64.acc32.srs"(%{{.*}}) <{shift = 8 : i32, sign = 1 : i32}>
func.func @explicit_rnd_only(%acc : vector<64xi32>) -> vector<64xi8>
    attributes {aie4.entry_modes = {rnd = 9 : i32}} {
  %0 = aievec.aie4.srs %acc {shift = 8 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>} : vector<64xi32> -> vector<64xi8>
  return %0 : vector<64xi8>
}

// -----

// v32 accepts shift 59.
// CHECK-LABEL: func.func @v32_shift59
// CHECK: "xllvm.intr.aie4.I512.v32.acc32.srs"(%{{.*}}) <{shift = 59 : i32, sign = 1 : i32}> : (vector<32xi32>) -> vector<32xi16>
func.func @v32_shift59(%acc : vector<32xi32>) -> vector<32xi16> {
  %0 = aievec.aie4.srs %acc {shift = 59 : i32, isSigned = true} : vector<32xi32> -> vector<32xi16>
  return %0 : vector<32xi16>
}


// -----

// Ambient SRS may observe user mode writes, even with aie4-assume-modes.
func.func private @llvm.aie4.set.ctrl.reg(i32, i32)
// CHECK-LABEL: func.func @ambient_write
// CHECK: call @llvm.aie4.set.ctrl.reg
// CHECK: "xllvm.intr.aie4.I512.v64.acc32.srs"
func.func @ambient_write(%acc: vector<64xi32>) -> vector<64xi8> {
  %reg = arith.constant 46 : i32
  %floor = arith.constant 0 : i32
  func.call @llvm.aie4.set.ctrl.reg(%reg, %floor) : (i32, i32) -> ()
  %y = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true} : vector<64xi32> -> vector<64xi8>
  return %y : vector<64xi8>
}

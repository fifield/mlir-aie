// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// aie4-assume-modes states the caller's modes for a function that carries no
// aie4.entry_modes (one not reached from an aie.core).
//
// RUN: aie-opt %s -convert-aievec-to-llvm="aie-target=aie4b aie4-assume-modes=rnd:9,sat:1" | FileCheck %s
// RUN: aie-opt %s -convert-aievec-to-llvm="aie-target=aie4b aie4-assume-modes=sat:1,rnd:9" | FileCheck %s

// No attribute: the option states the caller's modes.
// CHECK-LABEL: func.func @assumed
// CHECK: "xllvm.intr.aie4.I512.v64.acc32.srs"(%{{.*}}) <{shift = 12 : i32, sign = 1 : i32}>
// CHECK: "xllvm.intr.aie4.I512.v32.acc32.srs"(%{{.*}}) <{shift = 12 : i32, sign = 0 : i32}>
func.func @assumed(%a : vector<64xi32>, %b : vector<32xi32>) -> (vector<64xi8>, vector<32xi16>) {
  %0 = aievec.aie4.srs %a {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  %1 = aievec.aie4.srs %b {shift = 12 : i32, isSigned = false, saturation = #aievec.aie4_sat<saturate>} : vector<32xi32> -> vector<32xi16>
  return %0, %1 : vector<64xi8>, vector<32xi16>
}

// A pure internal closure is inspectable and preserves the assumption.
func.func private @pure_leaf(%v: i32) -> i32 {
  %one = arith.constant 1 : i32
  %result = arith.addi %v, %one : i32
  return %result : i32
}
func.func private @pure_middle(%v: i32) -> i32 {
  %result = func.call @pure_leaf(%v) : (i32) -> i32
  return %result : i32
}
// CHECK-LABEL: func.func @pure_helper_allowed
// CHECK: call @pure_middle
// CHECK: "xllvm.intr.aie4.I512.v64.acc32.srs"
func.func @pure_helper_allowed(%acc: vector<64xi32>, %v: i32) -> vector<64xi8> {
  %unused = func.call @pure_middle(%v) : (i32) -> i32
  %y = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  return %y : vector<64xi8>
}

// Known AIE4 non-writing intrinsics keep the same classification as in cores.
func.func private @llvm.aie4.event0()
// CHECK-LABEL: func.func @event_allowed
// CHECK: call @llvm.aie4.event0
// CHECK: "xllvm.intr.aie4.I512.v64.acc32.srs"
func.func @event_allowed(%acc: vector<64xi32>) -> vector<64xi8> {
  func.call @llvm.aie4.event0() : () -> ()
  %y = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  return %y : vector<64xi8>
}

// CHECK-LABEL: llvm.func @llvm_function_allowed
// CHECK: "xllvm.intr.aie4.I512.v64.acc32.srs"
llvm.func @llvm_function_allowed(%acc: vector<64xi32>) -> vector<64xi8> {
  %y = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
  llvm.return %y : vector<64xi8>
}

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --lower-vector-to-aievec="aie-target=aie4b" | FileCheck %s --implicit-check-not=aievec.aie4.srs_i32_i16
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" --convert-aievec-to-llvm="aie-target=aie4b" --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | FileCheck %s --check-prefix=LLVM --implicit-check-not=xllvm.intr.aie4.I512.v32.acc32.srs

// Source arithmetic must not inherit nondefault saturation or rounding modes.
func.func private @llvm.aie4.set.ctrl.reg(i32, i32)

// CHECK-LABEL: func.func @narrow_after_mode_change
// CHECK: call @llvm.aie4.set.ctrl.reg
// CHECK: call @llvm.aie4.set.ctrl.reg
// CHECK: arith.shrsi
// CHECK: arith.trunci
// LLVM-LABEL: llvm.func @narrow_after_mode_change
// LLVM: llvm.call @llvm.aie4.set.ctrl.reg
// LLVM: llvm.call @llvm.aie4.set.ctrl.reg
// LLVM: llvm.ashr
// LLVM: llvm.trunc
func.func @narrow_after_mode_change(%acc: vector<32xi32>) -> vector<32xi16> {
  %sat_reg = arith.constant 45 : i32
  %rnd_reg = arith.constant 46 : i32
  %mode = arith.constant 1 : i32
  func.call @llvm.aie4.set.ctrl.reg(%sat_reg, %mode) : (i32, i32) -> ()
  func.call @llvm.aie4.set.ctrl.reg(%rnd_reg, %mode) : (i32, i32) -> ()
  %shift = arith.constant dense<6> : vector<32xi32>
  %shifted = arith.shrsi %acc, %shift : vector<32xi32>
  %result = arith.trunci %shifted : vector<32xi32> to vector<32xi16>
  return %result : vector<32xi16>
}

// An opaque call may also change the modes.
func.func private @change_modes()

// CHECK-LABEL: func.func @narrow_after_call
// CHECK: call @change_modes
// CHECK: arith.shrsi
// CHECK: arith.trunci
// LLVM-LABEL: llvm.func @narrow_after_call
// LLVM: llvm.call @change_modes
// LLVM: llvm.ashr
// LLVM: llvm.trunc
func.func @narrow_after_call(%acc: vector<32xi32>) -> vector<32xi16> {
  func.call @change_modes() : () -> ()
  %shift = arith.constant dense<6> : vector<32xi32>
  %shifted = arith.shrsi %acc, %shift : vector<32xi32>
  %result = arith.trunci %shifted : vector<32xi32> to vector<32xi16>
  return %result : vector<32xi16>
}

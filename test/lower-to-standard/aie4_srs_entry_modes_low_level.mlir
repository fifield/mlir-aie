// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt --aie-standard-lowering --convert-aievec-to-llvm="aie-target=aie4b" --split-input-file %s | FileCheck %s

// Raw SRS reads ambient crRnd; only the independent crVecRnd is initialized.
// CHECK-LABEL: module @ambient_srs
// CHECK-LABEL: func.func @core_0_2()
// CHECK: arith.constant 24 : i32
// CHECK: call @llvm.aie4.set.ctrl.reg
// CHECK-NOT: call @llvm.aie4.set.ctrl.reg
// CHECK: call @raw_helper
// CHECK-NOT: call @llvm.aie4.set.ctrl.reg
// CHECK: return
module @ambient_srs {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    func.func private @raw_helper(%v: vector<64xi32>) -> vector<64xi8> {
      %raw = "xllvm.intr.aie4.I512.v64.acc32.srs"(%v) {shift = 1 : i32, sign = 1 : i32} : (vector<64xi32>) -> vector<64xi8>
      return %raw : vector<64xi8>
    }
    aie.core(%t) {
      %v = arith.constant dense<0> : vector<64xi32>
      %raw = func.call @raw_helper(%v) : (vector<64xi32>) -> vector<64xi8>
      aie.end
    }
  }
}

// -----

// Shift-zero UPS cannot overflow, for both attribute and operand spellings.
// Exact bf16 widening and unpack likewise impose no ambient-mode constraint.
// CHECK-LABEL: module @exact_widening
// CHECK: "xllvm.intr.aie4.acc32.v32.I512.ups"
// CHECK: call @llvm.aie4.acc32.v32.I512.ups
// CHECK: llvm.call_intrinsic "llvm.aie4.acc32.v32.I512.ups"
// CHECK: "xllvm.intr.aie4.v32bf16.to.v32accfloat"
// CHECK: "xllvm.intr.aie4.unpack.I512.I16.I8"
// CHECK: func.func @core_0_2() attributes {aie4.entry_modes = {rnd = 9 : i32, sat = 1 : i32}}
module @exact_widening {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    func.func private @llvm.aie4.acc32.v32.I512.ups(vector<32xi16>, i32, i32) -> vector<32xi32>
    func.func private @widen_helper() {
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      %v = arith.constant dense<0> : vector<32xi16>
      %f = arith.constant dense<0.0> : vector<32xbf16>
      %b = arith.constant dense<0> : vector<32xi8>
      %x = "xllvm.intr.aie4.acc32.v32.I512.ups"(%v) {shift = 0 : i32, sign = 1 : i32} : (vector<32xi16>) -> vector<32xi32>
      %y = func.call @llvm.aie4.acc32.v32.I512.ups(%v, %zero, %one) : (vector<32xi16>, i32, i32) -> vector<32xi32>
      %z = llvm.call_intrinsic "llvm.aie4.acc32.v32.I512.ups"(%v, %zero, %one) : (vector<32xi16>, i32, i32) -> vector<32xi32>
      %wide = "xllvm.intr.aie4.v32bf16.to.v32accfloat"(%f) : (vector<32xbf16>) -> vector<32xf32>
      %unpack = "xllvm.intr.aie4.unpack.I512.I16.I8"(%b) {sign = 1 : i32} : (vector<32xi8>) -> vector<32xi16>
      return
    }
    aie.core(%t) {
      %acc = arith.constant dense<0> : vector<64xi32>
      %explicit = aievec.aie4.srs %acc {shift = 1 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
      func.call @widen_helper() : () -> ()
      aie.end
    }
  }
}

//===- conv-f32-i32-aie4.mlir - AIE4 f32 -> i32 lowering -----------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" | FileCheck %s

// One instruction, and no control-register traffic: rounding comes from the
// ambient crRnd, which AIECoreToStandard sets at core entry. Setting it here
// instead would not survive -- Peano fuses a conversion that feeds a store
// into vst.conv and drops adjacent set.ctrl.reg calls.

// CHECK-LABEL: @conv_signed
// CHECK-NOT: xllvm.intr.aie4.set.ctrl.reg
// CHECK: "xllvm.intr.aie4.v32float.to.v32int32"(%{{.*}}) <{scale = 0 : i32, sign = 1 : i32}>
// CHECK-NOT: xllvm.intr.aie4.set.ctrl.reg
func.func @conv_signed(%a: vector<32xf32>) -> vector<32xi32> {
  %0 = aievec.aie4.conv_f32_i32 %a : vector<32xf32>, vector<32xi32>
  return %0 : vector<32xi32>
}

// -----

// CHECK-LABEL: @conv_unsigned
// CHECK: "xllvm.intr.aie4.v32float.to.v32int32"(%{{.*}}) <{scale = 0 : i32, sign = 0 : i32}>
func.func @conv_unsigned(%a: vector<32xf32>) -> vector<32xi32> {
  %0 = aievec.aie4.conv_f32_i32 %a {isSigned = false} : vector<32xf32>, vector<32xi32>
  return %0 : vector<32xi32>
}

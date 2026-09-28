// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --lower-vector-to-aievec="aie-target=aie4b" | FileCheck %s --implicit-check-not=aievec.aie4.fminmax --implicit-check-not=aievec.aie4.fcmp
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" --convert-aievec-to-llvm="aie-target=aie4b" --convert-vector-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | FileCheck %s --check-prefix=LLVM --implicit-check-not=xllvm.intr.aie4.vaddfloat

// Source IEEE operations must work even after a change to the ordering mode.
func.func private @llvm.aie4.set.ctrl.reg(i32, i32)
func.func private @change_modes()

// CHECK-LABEL: @after_mode_change
// CHECK: call @llvm.aie4.set.ctrl.reg
// CHECK: aievec.aie4.cmp
// CHECK: aievec.aie4.sel
// CHECK: return
// LLVM-LABEL: @after_mode_change
// LLVM: llvm.call @llvm.aie4.set.ctrl.reg
// LLVM: xllvm.intr.aie4.vadd32.512
// LLVM: xllvm.intr.aie4.vadd32.mask.512
// LLVM: llvm.return
func.func @after_mode_change(%a: vector<16xf32>, %b: vector<16xf32>) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xi1>) {
  %reg = arith.constant 47 : i32
  %mode = arith.constant 1 : i32
  func.call @llvm.aie4.set.ctrl.reg(%reg, %mode) : (i32, i32) -> ()
  %min = arith.minimumf %a, %b : vector<16xf32>
  %max = arith.maximumf %a, %b : vector<16xf32>
  %minnum = arith.minnumf %a, %b : vector<16xf32>
  %maxnum = arith.maxnumf %a, %b fastmath<nnan> : vector<16xf32>
  %cmp = arith.cmpf oge, %a, %b : vector<16xf32>
  return %min, %max, %minnum, %maxnum, %cmp : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xi1>
}

// Unknown calls and entry modes require the same protection.
// CHECK-LABEL: @after_call
// CHECK: call @change_modes
// CHECK: aievec.aie4.cmp
// CHECK: aievec.aie4.sel
// LLVM-LABEL: @after_call
// LLVM: llvm.call @change_modes
// LLVM: xllvm.intr.aie4.vadd32.1024
// LLVM: xllvm.intr.aie4.vadd32.mask.1024
func.func @after_call(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<32xf32> {
  func.call @change_modes() : () -> ()
  %min = arith.minimumf %a, %b : vector<32xf32>
  return %min : vector<32xf32>
}

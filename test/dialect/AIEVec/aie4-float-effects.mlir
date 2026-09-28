// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --cse | FileCheck %s
// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie4b" --cse | FileCheck %s --check-prefix=XLLVM

func.func private @change_modes()

// CHECK-LABEL: @native_mode_read
// CHECK: %[[M0:.*]] = aievec.aie4.fminmax
// CHECK: %[[C0:.*]] = aievec.aie4.fcmp
// CHECK: call @change_modes
// CHECK: %[[M1:.*]] = aievec.aie4.fminmax
// CHECK: %[[C1:.*]] = aievec.aie4.fcmp
// CHECK: return %[[M0]], %[[M1]], %[[C0]], %[[C1]]
// XLLVM-LABEL: @native_mode_read
// XLLVM: xllvm.intr.aie4.vaddfloat.512
// XLLVM: xllvm.intr.aie4.vaddfloat.512
// XLLVM: call @change_modes
// XLLVM: xllvm.intr.aie4.vaddfloat.512
// XLLVM: xllvm.intr.aie4.vaddfloat.512
func.func @native_mode_read(%a: vector<16xf32>, %b: vector<16xf32>) -> (vector<16xf32>, vector<16xf32>, ui32, ui32) {
  %m0 = aievec.aie4.fminmax %a, %b[kind = <min>] : vector<16xf32>
  %c0 = aievec.aie4.fcmp %a, %b[pred = <lt>] : vector<16xf32>, ui32
  func.call @change_modes() : () -> ()
  %m1 = aievec.aie4.fminmax %a, %b[kind = <min>] : vector<16xf32>
  %c1 = aievec.aie4.fcmp %a, %b[pred = <lt>] : vector<16xf32>, ui32
  return %m0, %m1, %c0, %c1 : vector<16xf32>, vector<16xf32>, ui32, ui32
}

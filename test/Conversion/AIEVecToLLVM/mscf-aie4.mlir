// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" | FileCheck %s

// f32 MSC: Float/Mode32x32 with zeroAcc cleared, as for the MAC form.
func.func @msc_ewise_f32(%a : vector<32xf32>, %b : vector<32xf32>,
                         %acc : vector<32xf32>) -> vector<32xf32> {
  %0 = aievec.aie4.msc_ewise_f32 %a, %b, %acc : vector<32xf32>
  return %0 : vector<32xf32>
}

// CHECK-LABEL: @msc_ewise_f32
// CHECK-SAME: %[[A:.*]]: vector<32xf32>, %[[B:.*]]: vector<32xf32>, %[[ACC:.*]]: vector<32xf32>
// CHECK: %[[R:.*]] = "xllvm.intr.aie4.I1024.I1024.ACC1024.fp.msc.ewise.conf"(
// CHECK-SAME: %[[A]], %[[B]], %[[ACC]]
// CHECK-SAME: amode = 1 : i32
// CHECK-SAME: bmode = 3 : i32
// CHECK-NOT: zeroAcc
// CHECK-SAME: (vector<32xf32>, vector<32xf32>, vector<32xf32>) -> vector<32xf32>

// -----

// bf16 x bf16 -> f32 MSC: Float/Mode16x16 selects the 32-lane form and both
// operand format bits select bf16.
func.func @msc_ewise_bf16_f32(%a : vector<32xbf16>, %b : vector<32xbf16>,
                              %acc : vector<32xf32>) -> vector<32xf32> {
  %0 = aievec.aie4.msc_ewise_bf16_f32 %a, %b, %acc : vector<32xbf16>, vector<32xf32>
  return %0 : vector<32xf32>
}

// CHECK-LABEL: @msc_ewise_bf16_f32
// CHECK-SAME: %[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>, %[[ACC:.*]]: vector<32xf32>
// CHECK: %[[R:.*]] = "xllvm.intr.aie4.bf16.I512.bf16.I512.ACC1024.fp.msc.ewise.conf"(
// CHECK-SAME: %[[A]], %[[B]], %[[ACC]]
// CHECK-SAME: amode = 1 : i32
// CHECK-SAME: bmode = 1 : i32
// CHECK-SAME: xIsBf16
// CHECK-SAME: yIsBf16
// CHECK-NOT: zeroAcc
// CHECK-SAME: (vector<32xbf16>, vector<32xbf16>, vector<32xf32>) -> vector<32xf32>

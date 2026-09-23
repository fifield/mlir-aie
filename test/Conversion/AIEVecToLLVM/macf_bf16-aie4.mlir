// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" | FileCheck %s

// f32 MAC: Float/Mode32x32 with zeroAcc cleared (conf 26).
func.func @mac_ewise_f32(%a : vector<32xf32>, %b : vector<32xf32>,
                         %acc : vector<32xf32>) -> vector<32xf32> {
  %0 = aievec.aie4.mac_ewise_f32 %a, %b, %acc : vector<32xf32>
  return %0 : vector<32xf32>
}

// CHECK-LABEL: @mac_ewise_f32
// CHECK-SAME: %[[A:.*]]: vector<32xf32>, %[[B:.*]]: vector<32xf32>, %[[ACC:.*]]: vector<32xf32>
// CHECK: %[[R:.*]] = "xllvm.intr.aie4.I1024.I1024.ACC1024.fp.mac.ewise.conf"(
// CHECK-SAME: %[[A]], %[[B]], %[[ACC]]
// CHECK-SAME: amode = 1 : i32
// CHECK-SAME: bmode = 3 : i32
// CHECK-NOT: zeroAcc
// CHECK-SAME: (vector<32xf32>, vector<32xf32>, vector<32xf32>) -> vector<32xf32>
// CHECK: return %[[R]] : vector<32xf32>

// -----

// bf16 -> f32 MUL: Float/Mode16x16 with both bf16 bits and zeroAcc (conf 107).
func.func @mul_ewise_bf16_f32(%a : vector<32xbf16>, %b : vector<32xbf16>)
    -> vector<32xf32> {
  %0 = aievec.aie4.mul_ewise_bf16_f32 %a, %b : vector<32xbf16>, vector<32xf32>
  return %0 : vector<32xf32>
}

// CHECK-LABEL: @mul_ewise_bf16_f32
// CHECK-SAME: %[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>
// CHECK: %[[R:.*]] = "xllvm.intr.aie4.bf16.I512.bf16.I512.ACC1024.fp.mul.ewise.conf"(
// CHECK-SAME: %[[A]], %[[B]]
// CHECK-SAME: amode = 1 : i32
// CHECK-SAME: bmode = 1 : i32
// CHECK-SAME: xIsBf16
// CHECK-SAME: yIsBf16
// CHECK-SAME: zeroAcc
// CHECK-SAME: (vector<32xbf16>, vector<32xbf16>) -> vector<32xf32>
// CHECK: return %[[R]] : vector<32xf32>

// -----

// bf16 -> f32 MAC: the same configuration with zeroAcc cleared (conf 106).
func.func @mac_ewise_bf16_f32(%a : vector<32xbf16>, %b : vector<32xbf16>,
                              %acc : vector<32xf32>) -> vector<32xf32> {
  %0 = aievec.aie4.mac_ewise_bf16_f32 %a, %b, %acc : vector<32xbf16>, vector<32xf32>
  return %0 : vector<32xf32>
}

// CHECK-LABEL: @mac_ewise_bf16_f32
// CHECK-SAME: %[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>, %[[ACC:.*]]: vector<32xf32>
// CHECK: %[[R:.*]] = "xllvm.intr.aie4.bf16.I512.bf16.I512.ACC1024.fp.mac.ewise.conf"(
// CHECK-SAME: %[[A]], %[[B]], %[[ACC]]
// CHECK-SAME: amode = 1 : i32
// CHECK-SAME: bmode = 1 : i32
// CHECK-SAME: xIsBf16
// CHECK-SAME: yIsBf16
// CHECK-NOT: zeroAcc
// CHECK-SAME: (vector<32xbf16>, vector<32xbf16>, vector<32xf32>) -> vector<32xf32>
// CHECK: return %[[R]] : vector<32xf32>

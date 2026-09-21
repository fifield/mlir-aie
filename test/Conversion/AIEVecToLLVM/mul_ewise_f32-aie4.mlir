// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" | FileCheck %s

// AIE4 elementwise f32 multiply lowers to a single XLLVM intrinsic op. The
// multiplier configuration word is carried as verified attributes
// (zeroAcc / amode=Float(1) / bmode=Mode32x32(3)) rather than an opaque i32.
func.func @mul_ewise_f32(%arg0 : vector<32xf32>, %arg1 : vector<32xf32>) -> vector<32xf32> {
  %0 = aievec.aie4.mul_ewise_f32 %arg0, %arg1 : vector<32xf32>
  return %0 : vector<32xf32>
}

// CHECK-LABEL: @mul_ewise_f32
// CHECK-SAME: %[[A:.*]]: vector<32xf32>, %[[B:.*]]: vector<32xf32>
// CHECK: %[[R:.*]] = "xllvm.intr.aie4.I1024.I1024.ACC1024.fp.mul.ewise.conf"(
// CHECK-SAME: %[[A]], %[[B]]
// CHECK-SAME: amode = 1 : i32
// CHECK-SAME: bmode = 3 : i32
// CHECK-SAME: zeroAcc
// CHECK-SAME: (vector<32xf32>, vector<32xf32>) -> vector<32xf32>
// CHECK: return %[[R]] : vector<32xf32>

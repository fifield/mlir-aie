// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-translate %s --mlir-to-llvmir | FileCheck %s

// Existing user symbols must not be reused as opaque native helpers. A renamed
// helper must still be shared by subsequent operations of the same shape/opsel.
llvm.func @__mlir_aie4_fporder_llvm.aie4.vaddfloat.512_36()
llvm.mlir.global external @__mlir_aie4_fporder_llvm.aie4.vaddfloat.512_36.1() : i32

// CHECK-LABEL: declare void @__mlir_aie4_fporder_llvm.aie4.vaddfloat.512_36()
// CHECK-LABEL: define { <16 x float>, i32 } @reuse_helper(
// CHECK: call { <16 x float>, i32 } @__mlir_aie4_fporder_llvm.aie4.vaddfloat.512_36.2(
// CHECK: call { <16 x float>, i32 } @__mlir_aie4_fporder_llvm.aie4.vaddfloat.512_36.2(
llvm.func @reuse_helper(%a: vector<16xf32>, %b: vector<16xf32>) -> !llvm.struct<(vector<16xf32>, i32)> {
  %first = "xllvm.intr.aie4.vaddfloat.512"(%a, %b) <{opsel = 36 : i32}> :
      (vector<16xf32>, vector<16xf32>) -> !llvm.struct<(vector<16xf32>, i32)>
  %v = llvm.extractvalue %first[0] : !llvm.struct<(vector<16xf32>, i32)>
  %second = "xllvm.intr.aie4.vaddfloat.512"(%v, %b) <{opsel = 36 : i32}> :
      (vector<16xf32>, vector<16xf32>) -> !llvm.struct<(vector<16xf32>, i32)>
  llvm.return %second : !llvm.struct<(vector<16xf32>, i32)>
}

// CHECK-LABEL: define internal { <16 x float>, i32 } @__mlir_aie4_fporder_llvm.aie4.vaddfloat.512_36.2(
// CHECK: call void @llvm.sideeffect()
// CHECK: call { <16 x float>, i32 } @llvm.aie4.vaddfloat.512(
// CHECK-SAME: i32 0, i32 36, i32 0)
// CHECK-NOT: define internal
// CHECK: attributes #{{[0-9]+}} = { noinline

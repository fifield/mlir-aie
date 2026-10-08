// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-translate %s -mlir-to-llvmir -split-input-file | FileCheck %s

// A stream read returns the same <32 x i32> type as aie.get_stream.
// CHECK-LABEL: define void @get_ss
llvm.func @get_ss(%port: i32, %out: !llvm.ptr) {
  // CHECK: call <32 x i32> @llvm.aie4.get.ss(i32 %{{[0-9]+}})
  %0 = "xllvm.intr.aie4.get.ss"(%port) : (i32) -> vector<32xi32>
  llvm.store %0, %out : vector<32xi32>, !llvm.ptr
  llvm.return
}
// CHECK: declare <32 x i32> @llvm.aie4.get.ss(i32)

// -----

// Reuse the declaration emitted by aie.get_stream lowering.
// CHECK: declare <32 x i32> @llvm.aie4.get.ss(i32)
// CHECK-NOT: declare
// CHECK-LABEL: define <32 x i32> @get_ss_with_get_stream
llvm.func @llvm.aie4.get.ss(i32) -> vector<32xi32>
llvm.func @get_ss_with_get_stream(%port: i32) -> vector<32xi32> {
  // CHECK: call <32 x i32> @llvm.aie4.get.ss(i32 %{{[0-9]+}})
  %0 = llvm.call @llvm.aie4.get.ss(%port) : (i32) -> vector<32xi32>
  // CHECK: call <32 x i32> @llvm.aie4.get.ss(i32 %{{[0-9]+}})
  %1 = "xllvm.intr.aie4.get.ss"(%port) : (i32) -> vector<32xi32>
  %2 = llvm.add %0, %1 : vector<32xi32>
  llvm.return %2 : vector<32xi32>
}
// CHECK-NOT: declare

// -----

// shift and sign become the trailing i32 arguments, in that order.
// CHECK-LABEL: define <64 x i8> @srs_v64
llvm.func @srs_v64(%acc: vector<64xi32>) -> vector<64xi8> {
  // CHECK: call <64 x i8> @llvm.aie4.I512.v64.acc32.srs(<64 x i32> %{{[0-9]+}}, i32 0, i32 0)
  %0 = "xllvm.intr.aie4.I512.v64.acc32.srs"(%acc) {shift = 0 : i32, sign = 0 : i32} : (vector<64xi32>) -> vector<64xi8>
  // CHECK: call <64 x i8> @llvm.aie4.I512.v64.acc32.srs(<64 x i32> %{{[0-9]+}}, i32 59, i32 1)
  %1 = "xllvm.intr.aie4.I512.v64.acc32.srs"(%acc) {shift = 59 : i32, sign = 1 : i32} : (vector<64xi32>) -> vector<64xi8>
  %2 = llvm.add %0, %1 : vector<64xi8>
  llvm.return %2 : vector<64xi8>
}
// CHECK: declare <64 x i8> @llvm.aie4.I512.v64.acc32.srs(<64 x i32>, i32, i32)

// -----

// Static and dynamic conf forms must share one intrinsic declaration.
// CHECK-LABEL: define <64 x i32> @mmac_dyn_conf
llvm.func @mmac_dyn_conf(%acc: vector<64xi32>, %a: vector<64xi8>, %first: i1) -> vector<64xi32> {
  // CHECK: %[[Z:[0-9]+]] = zext i1 %{{[0-9]+}} to i32
  // CHECK: %[[CONF:[0-9]+]] = or i32 776, %[[Z]]
  // CHECK: call <64 x i32> @llvm.aie4.I512.ACC2048.mmac(<64 x i32> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, i32 %[[CONF]])
  %c = llvm.mlir.constant(776 : i32) : i32
  %z = llvm.zext %first : i1 to i32
  %conf = llvm.or %c, %z : i32
  %0 = "xllvm.intr.aie4.I512.ACC2048.mmac.dyn_conf"(%acc, %a, %conf) : (vector<64xi32>, vector<64xi8>, i32) -> vector<64xi32>
  // CHECK: call <64 x i32> @llvm.aie4.I512.ACC2048.mmac(<64 x i32> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, i32 776)
  %1 = "xllvm.intr.aie4.I512.ACC2048.mmac"(%0, %a) {bmode = 1 : i32, xSign = 1 : i32, zSign = 1 : i32} : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
  llvm.return %1 : vector<64xi32>
}
// CHECK: declare <64 x i32> @llvm.aie4.I512.ACC2048.mmac(<64 x i32>, <64 x i8>, i32)
// CHECK-NOT: declare {{.*}}@llvm.aie4.I512.ACC2048.mmac

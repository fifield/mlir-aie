// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-translate %s -mlir-to-llvmir -split-input-file | FileCheck %s

// Dense MMAC, staging insert and commit intrinsics. Names and signatures
// follow utils/aie4_intrinsics_snapshot.yaml; the trailing i32 arguments are
// encoded from the typed attributes.

// CHECK-LABEL: define <64 x float> @mmac_bf16
llvm.func @mmac_bf16(%acc: vector<64xf32>, %a: vector<32xbf16>) -> vector<64xf32> {
  // CHECK: call <64 x float> @llvm.aie4.bf16.I512.ACC2048.mmac(<64 x float> %{{[0-9]+}}, <32 x bfloat> %{{[0-9]+}}, i32 0)
  %0 = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%acc, %a) : (vector<64xf32>, vector<32xbf16>) -> vector<64xf32>
  // CHECK: call <64 x float> @llvm.aie4.bf16.I512.ACC2048.mmac(<64 x float> %{{[0-9]+}}, <32 x bfloat> %{{[0-9]+}}, i32 1)
  %1 = "xllvm.intr.aie4.bf16.I512.ACC2048.mmac"(%0, %a) {zeroAcc} : (vector<64xf32>, vector<32xbf16>) -> vector<64xf32>
  llvm.return %1 : vector<64xf32>
}

// CHECK-LABEL: define <64 x i32> @mmac_int
llvm.func @mmac_int(%acc: vector<64xi32>, %a: vector<64xi8>) -> vector<64xi32> {
  // int8 x int8, both signed: bit 3, bit 8 and bit 9.
  // CHECK: call <64 x i32> @llvm.aie4.I512.ACC2048.mmac(<64 x i32> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, i32 776)
  %0 = "xllvm.intr.aie4.I512.ACC2048.mmac"(%acc, %a) {bmode = 1 : i32, xSign = 1 : i32, zSign = 1 : i32} : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
  // int4 x int4, both signed: bmode = 0 gives 0x300.
  // CHECK: call <64 x i32> @llvm.aie4.I512.ACC2048.mmac(<64 x i32> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, i32 768)
  %1 = "xllvm.intr.aie4.I512.ACC2048.mmac"(%0, %a) {bmode = 0 : i32, xSign = 1 : i32, zSign = 1 : i32} : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
  // Signed X only (bit 9) with zero_acc: 0x201.
  // CHECK: call <64 x i32> @llvm.aie4.I512.ACC2048.mmac(<64 x i32> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, i32 513)
  %2 = "xllvm.intr.aie4.I512.ACC2048.mmac"(%1, %a) {zeroAcc, bmode = 0 : i32, xSign = 1 : i32, zSign = 0 : i32} : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
  // Signed Z only (bit 8), int8: 0x108.
  // CHECK: call <64 x i32> @llvm.aie4.I512.ACC2048.mmac(<64 x i32> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, i32 264)
  %3 = "xllvm.intr.aie4.I512.ACC2048.mmac"(%2, %a) {bmode = 1 : i32, xSign = 0 : i32, zSign = 1 : i32} : (vector<64xi32>, vector<64xi8>) -> vector<64xi32>
  llvm.return %3 : vector<64xi32>
}

// CHECK-LABEL: define <64 x float> @mmac_fp8
llvm.func @mmac_fp8(%acc: vector<64xf32>, %a: vector<64xi8>) -> vector<64xf32> {
  // CHECK: call <64 x float> @llvm.aie4.fp8.I512.ACC2048.mmac(<64 x float> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, i32 0)
  %0 = "xllvm.intr.aie4.fp8.I512.ACC2048.mmac"(%acc, %a) {emode = 0 : i32} : (vector<64xf32>, vector<64xi8>) -> vector<64xf32>
  // emode = 2 in bits 11:10 with zero_acc: 0x801.
  // CHECK: call <64 x float> @llvm.aie4.fp8.I512.ACC2048.mmac(<64 x float> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, i32 2049)
  %1 = "xllvm.intr.aie4.fp8.I512.ACC2048.mmac"(%0, %a) {zeroAcc, emode = 2 : i32} : (vector<64xf32>, vector<64xi8>) -> vector<64xf32>
  // emode = 1 (expand_8x8_8x8) is bit 10 alone: 0x400.
  // CHECK: call <64 x float> @llvm.aie4.fp8.I512.ACC2048.mmac(<64 x float> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, i32 1024)
  %2 = "xllvm.intr.aie4.fp8.I512.ACC2048.mmac"(%1, %a) {emode = 1 : i32} : (vector<64xf32>, vector<64xi8>) -> vector<64xf32>
  llvm.return %2 : vector<64xf32>
}

// CHECK-LABEL: define void @staging
llvm.func @staging(%bf: vector<64xbf16>, %b8: vector<128xi8>, %i: i32) {
  // bf16 fixes crIntMMode = 2 (int8/bf16 data).
  // CHECK: call void @llvm.aie4.insert.staging.bf16.I1024(<64 x bfloat> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 2)
  "xllvm.intr.aie4.insert.staging.bf16.I1024"(%bf, %i) : (vector<64xbf16>, i32) -> ()
  // CHECK: call void @llvm.aie4.insert.staging.int.I1024(<128 x i8> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 3)
  "xllvm.intr.aie4.insert.staging.int.I1024"(%b8, %i) {mode = 3 : i32} : (vector<128xi8>, i32) -> ()
  // CHECK: call void @llvm.aie4.insert.staging.int.I1024(<128 x i8> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 0)
  "xllvm.intr.aie4.insert.staging.int.I1024"(%b8, %i) {mode = 0 : i32} : (vector<128xi8>, i32) -> ()
  // The fp8 insert has no mode argument.
  // CHECK: call void @llvm.aie4.insert.staging.fp8.I1024(<128 x i8> %{{[0-9]+}}, i32 %{{[0-9]+}})
  "xllvm.intr.aie4.insert.staging.fp8.I1024"(%b8, %i) : (vector<128xi8>, i32) -> ()
  // CHECK: call void @llvm.aie4.staging.to.matrix.I32768()
  "xllvm.intr.aie4.staging.to.matrix.I32768"() : () -> ()
  // CHECK: call void @llvm.aie4.staging.to.matrix.I4.I32768()
  "xllvm.intr.aie4.staging.to.matrix.I4.I32768"() : () -> ()
  // CHECK: call void @llvm.aie4.staging.to.matrix.FP32768()
  "xllvm.intr.aie4.staging.to.matrix.FP32768"() : () -> ()
  llvm.return
}

// CHECK-DAG: declare <64 x float> @llvm.aie4.bf16.I512.ACC2048.mmac(<64 x float>, <32 x bfloat>, i32)
// CHECK-DAG: declare <64 x i32> @llvm.aie4.I512.ACC2048.mmac(<64 x i32>, <64 x i8>, i32)
// CHECK-DAG: declare <64 x float> @llvm.aie4.fp8.I512.ACC2048.mmac(<64 x float>, <64 x i8>, i32)
// CHECK-DAG: declare void @llvm.aie4.insert.staging.bf16.I1024(<64 x bfloat>, i32, i32)
// CHECK-DAG: declare void @llvm.aie4.insert.staging.int.I1024(<128 x i8>, i32, i32)
// CHECK-DAG: declare void @llvm.aie4.insert.staging.fp8.I1024(<128 x i8>, i32)
// CHECK-DAG: declare void @llvm.aie4.staging.to.matrix.I32768()
// CHECK-DAG: declare void @llvm.aie4.staging.to.matrix.I4.I32768()
// CHECK-DAG: declare void @llvm.aie4.staging.to.matrix.FP32768()

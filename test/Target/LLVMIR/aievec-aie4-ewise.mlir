// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-translate %s -mlir-to-llvmir | FileCheck %s

// Check names and signatures for elementwise MUL/MAC/MSC, SRS/UPS, unpack,
// and bf16/f32 conversions. xllvm_verify.test verifies this with Peano opt.

// CHECK-LABEL: define <32 x float> @mul_ewise_f32
llvm.func @mul_ewise_f32(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<32xf32> {
  // CHECK: call <32 x float> @llvm.aie4.I1024.I1024.ACC1024.fp.mul.ewise.conf(
  // CHECK-SAME: <32 x float> %{{[0-9]+}}, <32 x float> %{{[0-9]+}}, i32 27)
  %0 = "xllvm.intr.aie4.I1024.I1024.ACC1024.fp.mul.ewise.conf"(%a, %b)
      <{amode = 1 : i32, bmode = 3 : i32, zeroAcc}> :
      (vector<32xf32>, vector<32xf32>) -> vector<32xf32>
  llvm.return %0 : vector<32xf32>
}

// CHECK-LABEL: define <32 x float> @mac_ewise_f32
llvm.func @mac_ewise_f32(%a: vector<32xf32>, %b: vector<32xf32>, %acc: vector<32xf32>) -> vector<32xf32> {
  // CHECK: call <32 x float> @llvm.aie4.I1024.I1024.ACC1024.fp.mac.ewise.conf(
  // CHECK-SAME: <32 x float> %{{[0-9]+}}, <32 x float> %{{[0-9]+}}, <32 x float> %{{[0-9]+}}, i32 26)
  %0 = "xllvm.intr.aie4.I1024.I1024.ACC1024.fp.mac.ewise.conf"(%a, %b, %acc)
      <{amode = 1 : i32, bmode = 3 : i32}> :
      (vector<32xf32>, vector<32xf32>, vector<32xf32>) -> vector<32xf32>
  llvm.return %0 : vector<32xf32>
}

// CHECK-LABEL: define <32 x float> @msc_ewise_f32
llvm.func @msc_ewise_f32(%a: vector<32xf32>, %b: vector<32xf32>, %acc: vector<32xf32>) -> vector<32xf32> {
  // CHECK: call <32 x float> @llvm.aie4.I1024.I1024.ACC1024.fp.msc.ewise.conf(
  // CHECK-SAME: <32 x float> %{{[0-9]+}}, <32 x float> %{{[0-9]+}}, <32 x float> %{{[0-9]+}}, i32 26)
  %0 = "xllvm.intr.aie4.I1024.I1024.ACC1024.fp.msc.ewise.conf"(%a, %b, %acc)
      <{amode = 1 : i32, bmode = 3 : i32}> :
      (vector<32xf32>, vector<32xf32>, vector<32xf32>) -> vector<32xf32>
  llvm.return %0 : vector<32xf32>
}

// CHECK-LABEL: define <32 x i32> @mac_ewise_i16_i32
llvm.func @mac_ewise_i16_i32(%a: vector<32xi16>, %b: vector<32xi16>, %acc: vector<32xi32>) -> vector<32xi32> {
  // CHECK: call <32 x i32> @llvm.aie4.I512.I512.ACC1024.mac.ewise.conf(
  // CHECK-SAME: <32 x i16> %{{[0-9]+}}, <32 x i16> %{{[0-9]+}}, <32 x i32> %{{[0-9]+}}, i32 776)
  %0 = "xllvm.intr.aie4.I512.I512.ACC1024.mac.ewise.conf"(%a, %b, %acc)
      <{amode = 0 : i32, bmode = 1 : i32, lhsSign = 1 : i32, rhsSign = 1 : i32}> :
      (vector<32xi16>, vector<32xi16>, vector<32xi32>) -> vector<32xi32>
  llvm.return %0 : vector<32xi32>
}

// CHECK-LABEL: define <32 x float> @mul_ewise_bf16_f32
llvm.func @mul_ewise_bf16_f32(%a: vector<32xbf16>, %b: vector<32xbf16>) -> vector<32xf32> {
  // CHECK: call <32 x float> @llvm.aie4.bf16.I512.bf16.I512.ACC1024.fp.mul.ewise.conf(
  // CHECK-SAME: <32 x bfloat> %{{[0-9]+}}, <32 x bfloat> %{{[0-9]+}}, i32 107)
  %0 = "xllvm.intr.aie4.bf16.I512.bf16.I512.ACC1024.fp.mul.ewise.conf"(%a, %b)
      <{amode = 1 : i32, bmode = 1 : i32, xIsBf16, yIsBf16, zeroAcc}> :
      (vector<32xbf16>, vector<32xbf16>) -> vector<32xf32>
  llvm.return %0 : vector<32xf32>
}

// CHECK-LABEL: define <32 x float> @mac_ewise_bf16_f32
llvm.func @mac_ewise_bf16_f32(%a: vector<32xbf16>, %b: vector<32xbf16>, %acc: vector<32xf32>) -> vector<32xf32> {
  // CHECK: call <32 x float> @llvm.aie4.bf16.I512.bf16.I512.ACC1024.fp.mac.ewise.conf(
  // CHECK-SAME: <32 x bfloat> %{{[0-9]+}}, <32 x bfloat> %{{[0-9]+}}, <32 x float> %{{[0-9]+}}, i32 106)
  %0 = "xllvm.intr.aie4.bf16.I512.bf16.I512.ACC1024.fp.mac.ewise.conf"(%a, %b, %acc)
      <{amode = 1 : i32, bmode = 1 : i32, xIsBf16, yIsBf16}> :
      (vector<32xbf16>, vector<32xbf16>, vector<32xf32>) -> vector<32xf32>
  llvm.return %0 : vector<32xf32>
}

// CHECK-LABEL: define <32 x float> @msc_ewise_bf16_f32
llvm.func @msc_ewise_bf16_f32(%a: vector<32xbf16>, %b: vector<32xbf16>, %acc: vector<32xf32>) -> vector<32xf32> {
  // CHECK: call <32 x float> @llvm.aie4.bf16.I512.bf16.I512.ACC1024.fp.msc.ewise.conf(
  // CHECK-SAME: <32 x bfloat> %{{[0-9]+}}, <32 x bfloat> %{{[0-9]+}}, <32 x float> %{{[0-9]+}}, i32 106)
  %0 = "xllvm.intr.aie4.bf16.I512.bf16.I512.ACC1024.fp.msc.ewise.conf"(%a, %b, %acc)
      <{amode = 1 : i32, bmode = 1 : i32, xIsBf16, yIsBf16}> :
      (vector<32xbf16>, vector<32xbf16>, vector<32xf32>) -> vector<32xf32>
  llvm.return %0 : vector<32xf32>
}

// CHECK-LABEL: define <32 x i16> @srs_i32_i16
llvm.func @srs_i32_i16(%acc: vector<32xi32>) -> vector<32xi16> {
  // CHECK: call <32 x i16> @llvm.aie4.I512.v32.acc32.srs(<32 x i32> %{{[0-9]+}}, i32 4, i32 1)
  %0 = "xllvm.intr.aie4.I512.v32.acc32.srs"(%acc) <{shift = 4 : i32, sign = 1 : i32}> :
      (vector<32xi32>) -> vector<32xi16>
  llvm.return %0 : vector<32xi16>
}

// CHECK-LABEL: define <32 x i32> @ups_i16_i32
llvm.func @ups_i16_i32(%src: vector<32xi16>) -> vector<32xi32> {
  // CHECK: call <32 x i32> @llvm.aie4.acc32.v32.I512.ups(<32 x i16> %{{[0-9]+}}, i32 0, i32 1)
  %0 = "xllvm.intr.aie4.acc32.v32.I512.ups"(%src) <{shift = 0 : i32, sign = 1 : i32}> :
      (vector<32xi16>) -> vector<32xi32>
  llvm.return %0 : vector<32xi32>
}

// CHECK-LABEL: define <32 x i16> @unpack_i8_i16
llvm.func @unpack_i8_i16(%src: vector<32xi8>) -> vector<32xi16> {
  // CHECK: call <32 x i16> @llvm.aie4.unpack.I512.I16.I8(<32 x i8> %{{[0-9]+}}, i32 1)
  %0 = "xllvm.intr.aie4.unpack.I512.I16.I8"(%src) <{sign = 1 : i32}> :
      (vector<32xi8>) -> vector<32xi16>
  llvm.return %0 : vector<32xi16>
}

// CHECK-LABEL: define <32 x float> @conv_bf16_f32
llvm.func @conv_bf16_f32(%src: vector<32xbf16>) -> vector<32xf32> {
  // CHECK: call <32 x float> @llvm.aie4.v32bf16.to.v32accfloat(<32 x bfloat> %{{[0-9]+}})
  %0 = "xllvm.intr.aie4.v32bf16.to.v32accfloat"(%src) :
      (vector<32xbf16>) -> vector<32xf32>
  llvm.return %0 : vector<32xf32>
}

// CHECK-LABEL: define <32 x bfloat> @conv_f32_bf16
llvm.func @conv_f32_bf16(%src: vector<32xf32>) -> vector<32xbf16> {
  // CHECK: call <32 x bfloat> @llvm.aie4.v32accfloat.to.v32bfloat16(<32 x float> %{{[0-9]+}})
  %0 = "xllvm.intr.aie4.v32accfloat.to.v32bfloat16"(%src) :
      (vector<32xf32>) -> vector<32xbf16>
  llvm.return %0 : vector<32xbf16>
}

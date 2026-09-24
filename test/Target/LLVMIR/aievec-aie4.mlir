// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-translate %s -mlir-to-llvmir -split-input-file | FileCheck %s

// Check LLVM intrinsic names, signatures, and operand order for all five
// supported vector shapes. Select uses sign = 0, opsel = 2, and fporder = 0.

// -- CMP / MIN-MAX (vaddNN.SIZE) --

// CHECK-LABEL: define { <64 x i8>, <2 x i32> } @vadd8_512
llvm.func @vadd8_512(%lhs: vector<64xi8>, %rhs: vector<64xi8>) -> !llvm.struct<(vector<64xi8>, vector<2xi32>)> {
    // CHECK: call { <64 x i8>, <2 x i32> } @llvm.aie4.vadd8.512(
    // CHECK-SAME: <64 x i8> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, i32 1, i32 13, i32 0)
    %0 = "xllvm.intr.aie4.vadd8.512"(%lhs, %rhs) <{opsel = 13 : i32, sign = 1 : i32}> :
        (vector<64xi8>, vector<64xi8>) -> !llvm.struct<(vector<64xi8>, vector<2xi32>)>
    llvm.return %0 : !llvm.struct<(vector<64xi8>, vector<2xi32>)>
}

// CHECK-LABEL: define { <32 x i16>, i32 } @vadd16_512
llvm.func @vadd16_512(%lhs: vector<32xi16>, %rhs: vector<32xi16>) -> !llvm.struct<(vector<32xi16>, i32)> {
    // CHECK: call { <32 x i16>, i32 } @llvm.aie4.vadd16.512(
    // CHECK-SAME: <32 x i16> %{{[0-9]+}}, <32 x i16> %{{[0-9]+}}, i32 0, i32 14, i32 0)
    %0 = "xllvm.intr.aie4.vadd16.512"(%lhs, %rhs) <{opsel = 14 : i32, sign = 0 : i32}> :
        (vector<32xi16>, vector<32xi16>) -> !llvm.struct<(vector<32xi16>, i32)>
    llvm.return %0 : !llvm.struct<(vector<32xi16>, i32)>
}

// CHECK-LABEL: define { <64 x i16>, <2 x i32> } @vadd16_1024
llvm.func @vadd16_1024(%lhs: vector<64xi16>, %rhs: vector<64xi16>) -> !llvm.struct<(vector<64xi16>, vector<2xi32>)> {
    // CHECK: call { <64 x i16>, <2 x i32> } @llvm.aie4.vadd16.1024(
    // CHECK-SAME: <64 x i16> %{{[0-9]+}}, <64 x i16> %{{[0-9]+}}, i32 1, i32 15, i32 0)
    %0 = "xllvm.intr.aie4.vadd16.1024"(%lhs, %rhs) <{opsel = 15 : i32, sign = 1 : i32}> :
        (vector<64xi16>, vector<64xi16>) -> !llvm.struct<(vector<64xi16>, vector<2xi32>)>
    llvm.return %0 : !llvm.struct<(vector<64xi16>, vector<2xi32>)>
}

// CHECK-LABEL: define { <16 x i32>, i32 } @vadd32_512
llvm.func @vadd32_512(%lhs: vector<16xi32>, %rhs: vector<16xi32>) -> !llvm.struct<(vector<16xi32>, i32)> {
    // CHECK: call { <16 x i32>, i32 } @llvm.aie4.vadd32.512(
    // CHECK-SAME: <16 x i32> %{{[0-9]+}}, <16 x i32> %{{[0-9]+}}, i32 0, i32 17, i32 0)
    %0 = "xllvm.intr.aie4.vadd32.512"(%lhs, %rhs) <{opsel = 17 : i32, sign = 0 : i32}> :
        (vector<16xi32>, vector<16xi32>) -> !llvm.struct<(vector<16xi32>, i32)>
    llvm.return %0 : !llvm.struct<(vector<16xi32>, i32)>
}

// CHECK-LABEL: define { <32 x i32>, i32 } @vadd32_1024
llvm.func @vadd32_1024(%lhs: vector<32xi32>, %rhs: vector<32xi32>) -> !llvm.struct<(vector<32xi32>, i32)> {
    // CHECK: call { <32 x i32>, i32 } @llvm.aie4.vadd32.1024(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}}, <32 x i32> %{{[0-9]+}}, i32 1, i32 13, i32 0)
    %0 = "xllvm.intr.aie4.vadd32.1024"(%lhs, %rhs) <{opsel = 13 : i32, sign = 1 : i32}> :
        (vector<32xi32>, vector<32xi32>) -> !llvm.struct<(vector<32xi32>, i32)>
    llvm.return %0 : !llvm.struct<(vector<32xi32>, i32)>
}

// CHECK-LABEL: declare { <64 x i8>, <2 x i32> } @llvm.aie4.vadd8.512(<64 x i8>, <64 x i8>, i32, i32, i32)
// CHECK-LABEL: declare { <32 x i16>, i32 } @llvm.aie4.vadd16.512(<32 x i16>, <32 x i16>, i32, i32, i32)
// CHECK-LABEL: declare { <64 x i16>, <2 x i32> } @llvm.aie4.vadd16.1024(<64 x i16>, <64 x i16>, i32, i32, i32)
// CHECK-LABEL: declare { <16 x i32>, i32 } @llvm.aie4.vadd32.512(<16 x i32>, <16 x i32>, i32, i32, i32)
// CHECK-LABEL: declare { <32 x i32>, i32 } @llvm.aie4.vadd32.1024(<32 x i32>, <32 x i32>, i32, i32, i32)

// -----

// -- SEL (vaddNN.mask.SIZE) --

// CHECK-LABEL: define <64 x i8> @vadd8_mask_512
llvm.func @vadd8_mask_512(%lhs: vector<64xi8>, %rhs: vector<64xi8>, %sel: vector<2xi32>) -> vector<64xi8> {
    // sign/opsel/fporder are pinned literals (0, 2, 0), not attributes.
    // CHECK: call <64 x i8> @llvm.aie4.vadd8.mask.512(
    // CHECK-SAME: <64 x i8> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, i32 0, i32 2, <2 x i32> %{{[0-9]+}}, i32 0)
    %0 = "xllvm.intr.aie4.vadd8.mask.512"(%lhs, %rhs, %sel) :
        (vector<64xi8>, vector<64xi8>, vector<2xi32>) -> vector<64xi8>
    llvm.return %0 : vector<64xi8>
}

// CHECK-LABEL: define <32 x i16> @vadd16_mask_512
llvm.func @vadd16_mask_512(%lhs: vector<32xi16>, %rhs: vector<32xi16>, %sel: i32) -> vector<32xi16> {
    // CHECK: call <32 x i16> @llvm.aie4.vadd16.mask.512(
    // CHECK-SAME: <32 x i16> %{{[0-9]+}}, <32 x i16> %{{[0-9]+}}, i32 0, i32 2, i32 %{{[0-9]+}}, i32 0)
    %0 = "xllvm.intr.aie4.vadd16.mask.512"(%lhs, %rhs, %sel) :
        (vector<32xi16>, vector<32xi16>, i32) -> vector<32xi16>
    llvm.return %0 : vector<32xi16>
}

// CHECK-LABEL: define <64 x i16> @vadd16_mask_1024
llvm.func @vadd16_mask_1024(%lhs: vector<64xi16>, %rhs: vector<64xi16>, %sel: vector<2xi32>) -> vector<64xi16> {
    // CHECK: call <64 x i16> @llvm.aie4.vadd16.mask.1024(
    // CHECK-SAME: <64 x i16> %{{[0-9]+}}, <64 x i16> %{{[0-9]+}}, i32 0, i32 2, <2 x i32> %{{[0-9]+}}, i32 0)
    %0 = "xllvm.intr.aie4.vadd16.mask.1024"(%lhs, %rhs, %sel) :
        (vector<64xi16>, vector<64xi16>, vector<2xi32>) -> vector<64xi16>
    llvm.return %0 : vector<64xi16>
}

// CHECK-LABEL: define <16 x i32> @vadd32_mask_512
llvm.func @vadd32_mask_512(%lhs: vector<16xi32>, %rhs: vector<16xi32>, %sel: i32) -> vector<16xi32> {
    // CHECK: call <16 x i32> @llvm.aie4.vadd32.mask.512(
    // CHECK-SAME: <16 x i32> %{{[0-9]+}}, <16 x i32> %{{[0-9]+}}, i32 0, i32 2, i32 %{{[0-9]+}}, i32 0)
    %0 = "xllvm.intr.aie4.vadd32.mask.512"(%lhs, %rhs, %sel) :
        (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
    llvm.return %0 : vector<16xi32>
}

// CHECK-LABEL: define <32 x i32> @vadd32_mask_1024
llvm.func @vadd32_mask_1024(%lhs: vector<32xi32>, %rhs: vector<32xi32>, %sel: i32) -> vector<32xi32> {
    // CHECK: call <32 x i32> @llvm.aie4.vadd32.mask.1024(
    // CHECK-SAME: <32 x i32> %{{[0-9]+}}, <32 x i32> %{{[0-9]+}}, i32 0, i32 2, i32 %{{[0-9]+}}, i32 0)
    %0 = "xllvm.intr.aie4.vadd32.mask.1024"(%lhs, %rhs, %sel) :
        (vector<32xi32>, vector<32xi32>, i32) -> vector<32xi32>
    llvm.return %0 : vector<32xi32>
}

// CHECK-LABEL: declare <64 x i8> @llvm.aie4.vadd8.mask.512(<64 x i8>, <64 x i8>, i32, i32, <2 x i32>, i32)
// CHECK-LABEL: declare <32 x i16> @llvm.aie4.vadd16.mask.512(<32 x i16>, <32 x i16>, i32, i32, i32, i32)
// CHECK-LABEL: declare <64 x i16> @llvm.aie4.vadd16.mask.1024(<64 x i16>, <64 x i16>, i32, i32, <2 x i32>, i32)
// CHECK-LABEL: declare <16 x i32> @llvm.aie4.vadd32.mask.512(<16 x i32>, <16 x i32>, i32, i32, i32, i32)
// CHECK-LABEL: declare <32 x i32> @llvm.aie4.vadd32.mask.1024(<32 x i32>, <32 x i32>, i32, i32, i32, i32)

// -----

// Check the names and signatures of all nine reduction intrinsics.

// CHECK-LABEL: define i32 @min_reduce_i1024
llvm.func @min_reduce_i1024(%src: vector<32xi32>, %seed: i32) -> i32 {
    // CHECK: call i32 @llvm.aie4.min.reduce.I1024(<32 x i32> %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie4.min.reduce.I1024"(%src, %seed) : (vector<32xi32>, i32) -> i32
    llvm.return %0 : i32
}

// CHECK-LABEL: define i32 @max_reduce_i1024
llvm.func @max_reduce_i1024(%src: vector<32xi32>, %seed: i32) -> i32 {
    // CHECK: call i32 @llvm.aie4.max.reduce.I1024(<32 x i32> %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie4.max.reduce.I1024"(%src, %seed) : (vector<32xi32>, i32) -> i32
    llvm.return %0 : i32
}

// CHECK-LABEL: define i32 @add_reduce_i1024
llvm.func @add_reduce_i1024(%src: vector<32xi32>, %seed: i32) -> i32 {
    // CHECK: call i32 @llvm.aie4.add.reduce.I1024(<32 x i32> %{{[0-9]+}}, i32 %{{[0-9]+}})
    %0 = "xllvm.intr.aie4.add.reduce.I1024"(%src, %seed) : (vector<32xi32>, i32) -> i32
    llvm.return %0 : i32
}

// CHECK-LABEL: define float @min_reduce_fp1024
llvm.func @min_reduce_fp1024(%src: vector<32xf32>) -> f32 {
    // CHECK: call float @llvm.aie4.min.reduce.FP1024(<32 x float> %{{[0-9]+}})
    %0 = "xllvm.intr.aie4.min.reduce.FP1024"(%src) : (vector<32xf32>) -> f32
    llvm.return %0 : f32
}

// CHECK-LABEL: define float @min_reduce_fp1024_scl
llvm.func @min_reduce_fp1024_scl(%src: vector<32xf32>, %seed: f32) -> f32 {
    // CHECK: call float @llvm.aie4.min.reduce.FP1024.scl(<32 x float> %{{[0-9]+}}, float %{{[0-9]+}})
    %0 = "xllvm.intr.aie4.min.reduce.FP1024.scl"(%src, %seed) : (vector<32xf32>, f32) -> f32
    llvm.return %0 : f32
}

// CHECK-LABEL: define float @max_reduce_fp1024
llvm.func @max_reduce_fp1024(%src: vector<32xf32>) -> f32 {
    // CHECK: call float @llvm.aie4.max.reduce.FP1024(<32 x float> %{{[0-9]+}})
    %0 = "xllvm.intr.aie4.max.reduce.FP1024"(%src) : (vector<32xf32>) -> f32
    llvm.return %0 : f32
}

// CHECK-LABEL: define float @max_reduce_fp1024_scl
llvm.func @max_reduce_fp1024_scl(%src: vector<32xf32>, %seed: f32) -> f32 {
    // CHECK: call float @llvm.aie4.max.reduce.FP1024.scl(<32 x float> %{{[0-9]+}}, float %{{[0-9]+}})
    %0 = "xllvm.intr.aie4.max.reduce.FP1024.scl"(%src, %seed) : (vector<32xf32>, f32) -> f32
    llvm.return %0 : f32
}

// CHECK-LABEL: define float @add_reduce_fp1024
llvm.func @add_reduce_fp1024(%src: vector<32xf32>) -> f32 {
    // CHECK: call float @llvm.aie4.add.reduce.FP1024(<32 x float> %{{[0-9]+}})
    %0 = "xllvm.intr.aie4.add.reduce.FP1024"(%src) : (vector<32xf32>) -> f32
    llvm.return %0 : f32
}

// CHECK-LABEL: define float @add_reduce_fp1024_scl
llvm.func @add_reduce_fp1024_scl(%src: vector<32xf32>, %seed: f32) -> f32 {
    // CHECK: call float @llvm.aie4.add.reduce.FP1024.scl(<32 x float> %{{[0-9]+}}, float %{{[0-9]+}})
    %0 = "xllvm.intr.aie4.add.reduce.FP1024.scl"(%src, %seed) : (vector<32xf32>, f32) -> f32
    llvm.return %0 : f32
}

// CHECK-LABEL: declare i32 @llvm.aie4.min.reduce.I1024(<32 x i32>, i32)
// CHECK-LABEL: declare i32 @llvm.aie4.max.reduce.I1024(<32 x i32>, i32)
// CHECK-LABEL: declare i32 @llvm.aie4.add.reduce.I1024(<32 x i32>, i32)
// CHECK-LABEL: declare float @llvm.aie4.min.reduce.FP1024(<32 x float>)
// CHECK-LABEL: declare float @llvm.aie4.min.reduce.FP1024.scl(<32 x float>, float)
// CHECK-LABEL: declare float @llvm.aie4.max.reduce.FP1024(<32 x float>)
// CHECK-LABEL: declare float @llvm.aie4.max.reduce.FP1024.scl(<32 x float>, float)
// CHECK-LABEL: declare float @llvm.aie4.add.reduce.FP1024(<32 x float>)
// CHECK-LABEL: declare float @llvm.aie4.add.reduce.FP1024.scl(<32 x float>, float)

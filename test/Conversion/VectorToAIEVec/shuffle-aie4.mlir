// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lower supported square transposes and i32 interleaves to AIE4 shuffles.
// Unsupported shapes, element types, and masks must remain generic.

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" -split-input-file | FileCheck %s
// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie4b" --convert-vector-to-aievec="aie-target=aie4b" -split-input-file | FileCheck %s

// CHECK-LABEL: @transpose_4x4_i32
func.func @transpose_4x4_i32(%a: vector<4x4xi32>) -> vector<4x4xi32> {
  // CHECK: vector.shape_cast %{{.*}} : vector<4x4xi32> to vector<16xi32>
  // CHECK: aievec.aie4.shuffle %{{.*}}[mode = <t32_4x4>] : vector<16xi32>
  // CHECK: vector.shape_cast %{{.*}} : vector<16xi32> to vector<4x4xi32>
  %0 = vector.transpose %a, [1, 0] : vector<4x4xi32> to vector<4x4xi32>
  return %0 : vector<4x4xi32>
}

// -----

// CHECK-LABEL: @transpose_8x8_i8
func.func @transpose_8x8_i8(%a: vector<8x8xi8>) -> vector<8x8xi8> {
  // CHECK: vector.shape_cast %{{.*}} : vector<8x8xi8> to vector<64xi8>
  // CHECK: aievec.aie4.shuffle %{{.*}}[mode = <t8_8x8>] : vector<64xi8>
  // CHECK: vector.shape_cast %{{.*}} : vector<64xi8> to vector<8x8xi8>
  %0 = vector.transpose %a, [1, 0] : vector<8x8xi8> to vector<8x8xi8>
  return %0 : vector<8x8xi8>
}

// -----

// This non-square transpose has no lowering in the supported mode subset.
// CHECK-LABEL: @transpose_non_square_not_lowered
func.func @transpose_non_square_not_lowered(%a: vector<2x8xi32>) -> vector<8x2xi32> {
  // CHECK: vector.transpose
  // CHECK-NOT: aievec.aie4.shuffle
  %0 = vector.transpose %a, [1, 0] : vector<2x8xi32> to vector<8x2xi32>
  return %0 : vector<8x2xi32>
}

// -----

// CHECK-LABEL: @transpose_4x4_i16_wrong_width_not_lowered
func.func @transpose_4x4_i16_wrong_width_not_lowered(%a: vector<4x4xi16>) -> vector<4x4xi16> {
  // CHECK: vector.transpose
  // CHECK-NOT: aievec.aie4.shuffle
  %0 = vector.transpose %a, [1, 0] : vector<4x4xi16> to vector<4x4xi16>
  return %0 : vector<4x4xi16>
}

// -----

// CHECK-LABEL: @shuffle_zip_lo
func.func @shuffle_zip_lo(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // CHECK: aievec.aie4.shuffle %{{.*}}, %{{.*}}[mode = <t32_2x16_lo>] : vector<16xi32>
  %0 = vector.shuffle %a, %b [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23]
      : vector<16xi32>, vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

// CHECK-LABEL: @shuffle_zip_hi
func.func @shuffle_zip_hi(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // CHECK: aievec.aie4.shuffle %{{.*}}, %{{.*}}[mode = <t32_2x16_hi>] : vector<16xi32>
  %0 = vector.shuffle %a, %b [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31]
      : vector<16xi32>, vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

// CHECK-LABEL: @shuffle_non_zip_mask_not_lowered
func.func @shuffle_non_zip_mask_not_lowered(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // CHECK: vector.shuffle
  // CHECK-NOT: aievec.aie4.shuffle
  %0 = vector.shuffle %a, %b [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]
      : vector<16xi32>, vector<16xi32>
  return %0 : vector<16xi32>
}

// -----

// CHECK-LABEL: @transpose_4x4_f32
func.func @transpose_4x4_f32(%a: vector<4x4xf32>) -> vector<4x4xf32> {
  // CHECK: vector.shape_cast %{{.*}} : vector<4x4xf32> to vector<16xf32>
  // CHECK: aievec.aie4.shuffle %{{.*}}[mode = <t32_4x4>] : vector<16xf32>
  // CHECK: vector.shape_cast %{{.*}} : vector<16xf32> to vector<4x4xf32>
  %0 = vector.transpose %a, [1, 0] : vector<4x4xf32> to vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: @transpose_fp8_not_lowered
func.func @transpose_fp8_not_lowered(%a: vector<8x8xf8E4M3FN>) -> vector<8x8xf8E4M3FN> {
  // CHECK-NOT: aievec.aie4.shuffle
  // CHECK: vector.transpose %{{.*}}, [1, 0] : vector<8x8xf8E4M3FN> to vector<8x8xf8E4M3FN>
  // CHECK-NOT: aievec.aie4.shuffle
  // CHECK: return
  %0 = vector.transpose %a, [1, 0] : vector<8x8xf8E4M3FN> to vector<8x8xf8E4M3FN>
  return %0 : vector<8x8xf8E4M3FN>
}

// -----

// CHECK-LABEL: @transpose_scalable_not_lowered
func.func @transpose_scalable_not_lowered(%a: vector<4x[4]xi32>) -> vector<[4]x4xi32> {
  // CHECK-NOT: aievec.aie4.shuffle
  // CHECK: vector.transpose %{{.*}}, [1, 0] : vector<4x[4]xi32> to vector<[4]x4xi32>
  // CHECK-NOT: aievec.aie4.shuffle
  // CHECK: return
  %0 = vector.transpose %a, [1, 0] : vector<4x[4]xi32> to vector<[4]x4xi32>
  return %0 : vector<[4]x4xi32>
}

// -----

// CHECK-LABEL: @transpose_signed_not_lowered
func.func @transpose_signed_not_lowered(%a: vector<4x4xsi32>) -> vector<4x4xsi32> {
  // CHECK-NOT: aievec.aie4.shuffle
  // CHECK: vector.transpose %{{.*}}, [1, 0] : vector<4x4xsi32> to vector<4x4xsi32>
  // CHECK-NOT: aievec.aie4.shuffle
  // CHECK: return
  %0 = vector.transpose %a, [1, 0] : vector<4x4xsi32> to vector<4x4xsi32>
  return %0 : vector<4x4xsi32>
}

// -----

// CHECK-LABEL: @transpose_unsigned_not_lowered
func.func @transpose_unsigned_not_lowered(%a: vector<8x8xui8>) -> vector<8x8xui8> {
  // CHECK-NOT: aievec.aie4.shuffle
  // CHECK: vector.transpose %{{.*}}, [1, 0] : vector<8x8xui8> to vector<8x8xui8>
  // CHECK-NOT: aievec.aie4.shuffle
  // CHECK: return
  %0 = vector.transpose %a, [1, 0] : vector<8x8xui8> to vector<8x8xui8>
  return %0 : vector<8x8xui8>
}

// -----

// CHECK-LABEL: @shuffle_si32_not_lowered
func.func @shuffle_si32_not_lowered(%a: vector<16xsi32>, %b: vector<16xsi32>) -> vector<16xsi32> {
  // CHECK-NOT: aievec.aie4.shuffle
  // CHECK: vector.shuffle
  // CHECK-NOT: aievec.aie4.shuffle
  // CHECK: return
  %0 = vector.shuffle %a, %b [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23]
      : vector<16xsi32>, vector<16xsi32>
  return %0 : vector<16xsi32>
}

// -----

// CHECK-LABEL: @shuffle_ui32_not_lowered
func.func @shuffle_ui32_not_lowered(%a: vector<16xui32>, %b: vector<16xui32>) -> vector<16xui32> {
  // CHECK-NOT: aievec.aie4.shuffle
  // CHECK: vector.shuffle
  // CHECK-NOT: aievec.aie4.shuffle
  // CHECK: return
  %0 = vector.shuffle %a, %b [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23]
      : vector<16xui32>, vector<16xui32>
  return %0 : vector<16xui32>
}

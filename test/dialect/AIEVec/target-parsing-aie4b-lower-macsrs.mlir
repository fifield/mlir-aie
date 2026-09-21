//
// target-parsing-aie4b-lower-macsrs.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Fuse widened integer products into MAC; keep shift/truncate arithmetic
// independent of the mutable SRS rounding and saturation modes.
//
// RUN: aie-opt %s -split-input-file --lower-vector-to-aievec="aie-target=aie4b" | FileCheck %s --implicit-check-not=aievec.aie4.srs_i32_i16
// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie4b" | FileCheck %s --implicit-check-not=aievec.aie4.srs_i32_i16

// Fuse the i16 sign extensions, i32 multiply, and add into one native MAC.
//
// CHECK-LABEL: func.func @mac_v32i16_acc_v32i32
// CHECK: aievec.aie4.mac_ewise_i16_i32 %arg0, %arg1, %arg2 : vector<32xi16>, vector<32xi32>
// CHECK-NOT: arith.muli
// CHECK-NOT: arith.addi
// CHECK-NOT: arith.extsi
func.func @mac_v32i16_acc_v32i32(%a : vector<32xi16>, %b : vector<32xi16>,
                                 %acc : vector<32xi32>) -> vector<32xi32> {
  %ea = arith.extsi %a : vector<32xi16> to vector<32xi32>
  %eb = arith.extsi %b : vector<32xi16> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %s = arith.addi %acc, %p : vector<32xi32>
  return %s : vector<32xi32>
}

// -----

// `arith.addi` is commutative, so the product may appear on either side.
// CHECK-LABEL: func.func @mac_operands_swapped
// CHECK: aievec.aie4.mac_ewise_i16_i32 %arg0, %arg1, %arg2 : vector<32xi16>, vector<32xi32>
func.func @mac_operands_swapped(%a : vector<32xi16>, %b : vector<32xi16>,
                                %acc : vector<32xi32>) -> vector<32xi32> {
  %ea = arith.extsi %a : vector<32xi16> to vector<32xi32>
  %eb = arith.extsi %b : vector<32xi16> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %s = arith.addi %p, %acc : vector<32xi32>
  return %s : vector<32xi32>
}

// -----

// A plain add must remain legal when no MAC pattern matches.
// CHECK-LABEL: func.func @plain_addi_untouched
// CHECK: arith.addi
// CHECK-NOT: aievec.aie4.mac_ewise_i16_i32
func.func @plain_addi_untouched(%a : vector<32xi32>, %b : vector<32xi32>) -> vector<32xi32> {
  %0 = arith.addi %a, %b : vector<32xi32>
  return %0 : vector<32xi32>
}

// -----

// i32 operands without i16 sign extensions do not match the native MAC.
// CHECK-LABEL: func.func @mac_not_widened_from_i16
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK-NOT: aievec.aie4.mac_ewise_i16_i32
func.func @mac_not_widened_from_i16(%a : vector<32xi32>, %b : vector<32xi32>,
                                    %acc : vector<32xi32>) -> vector<32xi32> {
  %p = arith.muli %a, %b : vector<32xi32>
  %s = arith.addi %acc, %p : vector<32xi32>
  return %s : vector<32xi32>
}

// -----

// The i16 MAC requires 32 lanes; leave 16-lane arithmetic unchanged.
// CHECK-LABEL: func.func @mac_v16_untouched
// CHECK: arith.muli
// CHECK-NOT: aievec.aie4.mac_ewise_i16_i32
func.func @mac_v16_untouched(%a : vector<16xi16>, %b : vector<16xi16>,
                             %acc : vector<16xi32>) -> vector<16xi32> {
  %ea = arith.extsi %a : vector<16xi16> to vector<16xi32>
  %eb = arith.extsi %b : vector<16xi16> to vector<16xi32>
  %p = arith.muli %ea, %eb : vector<16xi32>
  %s = arith.addi %acc, %p : vector<16xi32>
  return %s : vector<16xi32>
}

// -----

// Preserve floor rounding and wrapping without assuming the current SRS modes.
// CHECK-LABEL: func.func @srs_shift_narrow
// CHECK: arith.shrsi
// CHECK: arith.trunci
func.func @srs_shift_narrow(%acc : vector<32xi32>) -> vector<32xi16> {
  %sh = arith.constant dense<4> : vector<32xi32>
  %r = arith.shrsi %acc, %sh : vector<32xi32>
  %t = arith.trunci %r : vector<32xi32> to vector<32xi16>
  return %t : vector<32xi16>
}

// -----

// A zero shift may fold away; narrowing must remain arithmetic.
// CHECK-LABEL: func.func @srs_shift_zero
// CHECK: arith.trunci
func.func @srs_shift_zero(%acc : vector<32xi32>) -> vector<32xi16> {
  %sh = arith.constant dense<0> : vector<32xi32>
  %r = arith.shrsi %acc, %sh : vector<32xi32>
  %t = arith.trunci %r : vector<32xi32> to vector<32xi16>
  return %t : vector<32xi16>
}

// -----

// Clamped narrowing must also remain independent of the SRS modes.
// CHECK-LABEL: func.func @clamped_narrow_is_not_srs
// CHECK: arith.trunci
// CHECK-NOT: aievec.aie4.srs_i32_i16
func.func @clamped_narrow_is_not_srs(%acc : vector<32xi32>) -> vector<32xi16> {
  %sh = arith.constant dense<4> : vector<32xi32>
  %lo = arith.constant dense<-32768> : vector<32xi32>
  %hi = arith.constant dense<32767> : vector<32xi32>
  %r = arith.shrsi %acc, %sh : vector<32xi32>
  %m = arith.maxsi %r, %lo : vector<32xi32>
  %n = arith.minsi %m, %hi : vector<32xi32>
  %t = arith.trunci %n : vector<32xi32> to vector<32xi16>
  return %t : vector<32xi16>
}

// -----

// Preserve dynamic shifts and narrowing as arithmetic.
// CHECK-LABEL: func.func @srs_dynamic_shift_untouched
// CHECK: arith.trunci
// CHECK-NOT: aievec.aie4.srs_i32_i16
func.func @srs_dynamic_shift_untouched(%acc : vector<32xi32>,
                                       %sh : vector<32xi32>) -> vector<32xi16> {
  %r = arith.shrsi %acc, %sh : vector<32xi32>
  %t = arith.trunci %r : vector<32xi32> to vector<32xi16>
  return %t : vector<32xi16>
}

// -----

// MAC fusion remains enabled while its shift/truncate consumers stay generic.
// CHECK-LABEL: func.func @mac_then_srs
// CHECK: %[[M:.*]] = aievec.aie4.mac_ewise_i16_i32 %arg0, %arg1, %arg2
// CHECK: arith.shrsi %[[M]]
// CHECK: arith.trunci
func.func @mac_then_srs(%a : vector<32xi16>, %b : vector<32xi16>,
                        %acc : vector<32xi32>) -> vector<32xi16> {
  %ea = arith.extsi %a : vector<32xi16> to vector<32xi32>
  %eb = arith.extsi %b : vector<32xi16> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %s = arith.addi %acc, %p : vector<32xi32>
  %sh = arith.constant dense<6> : vector<32xi32>
  %r = arith.shrsi %s, %sh : vector<32xi32>
  %t = arith.trunci %r : vector<32xi32> to vector<32xi16>
  return %t : vector<32xi16>
}

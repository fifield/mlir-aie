// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native int8 widening and MAC lowering, including signedness and shared
// extensions. Canonicalization folds successive sign extensions from i8 to
// i16 to i32 into a direct i8 -> i32 extension.
//
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" | FileCheck %s
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" \
// RUN:   --convert-aievec-to-llvm="aie-target=aie4b" | FileCheck %s --check-prefix=LLVM

// An int8 MAC chain widens each operand to i16 exactly, then uses the i16 VMAC.
// CHECK-LABEL: @mac_i8
// CHECK-DAG:   %[[A:.*]] = aievec.aie4.unpack_i8_i16 %arg0 : vector<32xi8>, vector<32xi16>
// CHECK-DAG:   %[[B:.*]] = aievec.aie4.unpack_i8_i16 %arg1 : vector<32xi8>, vector<32xi16>
// CHECK:       aievec.aie4.mac_ewise_i16_i32 %[[A]], %[[B]], %arg2
// CHECK-NOT:   arith.muli
// LLVM-LABEL:  @mac_i8
// LLVM-DAG:    "xllvm.intr.aie4.unpack.I512.I16.I8"(%arg0) <{sign = 1 : i32}> : (vector<32xi8>) -> vector<32xi16>
// LLVM-DAG:    "xllvm.intr.aie4.unpack.I512.I16.I8"(%arg1) <{sign = 1 : i32}> : (vector<32xi8>) -> vector<32xi16>
// LLVM:        "xllvm.intr.aie4.I512.I512.ACC1024.mac.ewise.conf"
func.func @mac_i8(%a: vector<32xi8>, %b: vector<32xi8>, %acc: vector<32xi32>) -> vector<32xi32> {
  %ea = arith.extsi %a : vector<32xi8> to vector<32xi32>
  %eb = arith.extsi %b : vector<32xi8> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %r = arith.addi %acc, %p : vector<32xi32>
  return %r : vector<32xi32>
}

// Mixed widths: only the i8 operand is unpacked.
// CHECK-LABEL: @mac_i8_i16
// CHECK:       %[[A:.*]] = aievec.aie4.unpack_i8_i16 %arg0 : vector<32xi8>, vector<32xi16>
// CHECK:       aievec.aie4.mac_ewise_i16_i32 %[[A]], %arg1, %arg2
func.func @mac_i8_i16(%a: vector<32xi8>, %b: vector<32xi16>, %acc: vector<32xi32>) -> vector<32xi32> {
  %ea = arith.extsi %a : vector<32xi8> to vector<32xi32>
  %eb = arith.extsi %b : vector<32xi16> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %r = arith.addi %p, %acc : vector<32xi32>
  return %r : vector<32xi32>
}

// Unsigned and mixed-sign int8 MACs unpack each operand with its own
// signedness; both i8 ranges fit in signed i16.
// CHECK-LABEL: @mac_u8_u8
// CHECK-DAG:   %[[A:.*]] = aievec.aie4.unpack_i8_i16 %arg0 {isSigned = false} : vector<32xi8>, vector<32xi16>
// CHECK-DAG:   %[[B:.*]] = aievec.aie4.unpack_i8_i16 %arg1 {isSigned = false} : vector<32xi8>, vector<32xi16>
// CHECK:       aievec.aie4.mac_ewise_i16_i32 %[[A]], %[[B]], %arg2
// CHECK-NOT:   arith.muli
// CHECK-NOT:   arith.extui
// LLVM-LABEL:  @mac_u8_u8
// LLVM-DAG:    "xllvm.intr.aie4.unpack.I512.I16.I8"(%arg0) <{sign = 0 : i32}>
// LLVM-DAG:    "xllvm.intr.aie4.unpack.I512.I16.I8"(%arg1) <{sign = 0 : i32}>
// LLVM:        "xllvm.intr.aie4.I512.I512.ACC1024.mac.ewise.conf"
func.func @mac_u8_u8(%a: vector<32xi8>, %b: vector<32xi8>, %acc: vector<32xi32>) -> vector<32xi32> {
  %ea = arith.extui %a : vector<32xi8> to vector<32xi32>
  %eb = arith.extui %b : vector<32xi8> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %r = arith.addi %acc, %p : vector<32xi32>
  return %r : vector<32xi32>
}

// CHECK-LABEL: @mac_s8_u8
// CHECK-DAG:   %[[A:.*]] = aievec.aie4.unpack_i8_i16 %arg0 : vector<32xi8>, vector<32xi16>
// CHECK-DAG:   %[[B:.*]] = aievec.aie4.unpack_i8_i16 %arg1 {isSigned = false} : vector<32xi8>, vector<32xi16>
// CHECK:       aievec.aie4.mac_ewise_i16_i32 %[[A]], %[[B]], %arg2
// CHECK-NOT:   arith.muli
// LLVM-LABEL:  @mac_s8_u8
// LLVM-DAG:    "xllvm.intr.aie4.unpack.I512.I16.I8"(%arg0) <{sign = 1 : i32}>
// LLVM-DAG:    "xllvm.intr.aie4.unpack.I512.I16.I8"(%arg1) <{sign = 0 : i32}>
func.func @mac_s8_u8(%a: vector<32xi8>, %b: vector<32xi8>, %acc: vector<32xi32>) -> vector<32xi32> {
  %ea = arith.extsi %a : vector<32xi8> to vector<32xi32>
  %eb = arith.extui %b : vector<32xi8> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %r = arith.addi %acc, %p : vector<32xi32>
  return %r : vector<32xi32>
}

// CHECK-LABEL: @mac_u8_s8
// CHECK-DAG:   %[[A:.*]] = aievec.aie4.unpack_i8_i16 %arg0 {isSigned = false} : vector<32xi8>, vector<32xi16>
// CHECK-DAG:   %[[B:.*]] = aievec.aie4.unpack_i8_i16 %arg1 : vector<32xi8>, vector<32xi16>
// CHECK:       aievec.aie4.mac_ewise_i16_i32 %[[A]], %[[B]], %arg2
// CHECK-NOT:   arith.muli
// LLVM-LABEL:  @mac_u8_s8
// LLVM-DAG:    "xllvm.intr.aie4.unpack.I512.I16.I8"(%arg0) <{sign = 0 : i32}>
// LLVM-DAG:    "xllvm.intr.aie4.unpack.I512.I16.I8"(%arg1) <{sign = 1 : i32}>
func.func @mac_u8_s8(%a: vector<32xi8>, %b: vector<32xi8>, %acc: vector<32xi32>) -> vector<32xi32> {
  %ea = arith.extui %a : vector<32xi8> to vector<32xi32>
  %eb = arith.extsi %b : vector<32xi8> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %r = arith.addi %p, %acc : vector<32xi32>
  return %r : vector<32xi32>
}

// u8 x i16: the i16 operand is read directly.
// CHECK-LABEL: @mac_u8_i16
// CHECK:       %[[A:.*]] = aievec.aie4.unpack_i8_i16 %arg0 {isSigned = false} : vector<32xi8>, vector<32xi16>
// CHECK:       aievec.aie4.mac_ewise_i16_i32 %[[A]], %arg1, %arg2
func.func @mac_u8_i16(%a: vector<32xi8>, %b: vector<32xi16>, %acc: vector<32xi32>) -> vector<32xi32> {
  %ea = arith.extui %a : vector<32xi8> to vector<32xi32>
  %eb = arith.extsi %b : vector<32xi16> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %r = arith.addi %acc, %p : vector<32xi32>
  return %r : vector<32xi32>
}

// Unsigned i16 does not fit in the signed 16x16 mode: no MAC; each extension
// becomes a standalone unsigned UPS and the multiply stays generic.
// CHECK-LABEL: @mac_u16_not_fused
// CHECK-NOT:   aievec.aie4.mac_ewise_i16_i32
// CHECK:       arith.muli
func.func @mac_u16_not_fused(%a: vector<32xi16>, %b: vector<32xi16>, %acc: vector<32xi32>) -> vector<32xi32> {
  %ea = arith.extui %a : vector<32xi16> to vector<32xi32>
  %eb = arith.extui %b : vector<32xi16> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %r = arith.addi %acc, %p : vector<32xi32>
  return %r : vector<32xi32>
}

// A use outside the MAC needs standalone widening; the MAC still consumes
// the narrow source through unpack.
// CHECK-LABEL: @mac_i8_multiuse
// CHECK-DAG:   %[[A:.*]] = aievec.aie4.unpack_i8_i16 %arg0 : vector<32xi8>, vector<32xi16>
// CHECK-DAG:   %[[EA:.*]] = aievec.aie4.ups_i16_i32 %{{.*}} {shift = 0 : i32} : vector<32xi16>, vector<32xi32>
// CHECK-DAG:   %[[B:.*]] = aievec.aie4.unpack_i8_i16 %arg1 : vector<32xi8>, vector<32xi16>
// CHECK:       %[[R:.*]] = aievec.aie4.mac_ewise_i16_i32 %[[A]], %[[B]], %arg2
// CHECK-NOT:   arith.extsi
// CHECK-NOT:   arith.muli
// CHECK:       return %[[R]], %[[EA]]
func.func @mac_i8_multiuse(%a: vector<32xi8>, %b: vector<32xi8>, %acc: vector<32xi32>) -> (vector<32xi32>, vector<32xi32>) {
  %ea = arith.extsi %a : vector<32xi8> to vector<32xi32>
  %eb = arith.extsi %b : vector<32xi8> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %r = arith.addi %acc, %p : vector<32xi32>
  return %r, %ea : vector<32xi32>, vector<32xi32>
}

// Shared i16 extensions need standalone UPS; the MAC reads i16 directly.
// CHECK-LABEL: @mac_i16_multiuse
// CHECK:       %[[EA:.*]] = aievec.aie4.ups_i16_i32 %arg0 {shift = 0 : i32} : vector<32xi16>, vector<32xi32>
// CHECK:       %[[R:.*]] = aievec.aie4.mac_ewise_i16_i32 %arg0, %arg1, %arg2
// CHECK-NOT:   arith.extsi
// CHECK:       return %[[R]], %[[EA]]
func.func @mac_i16_multiuse(%a: vector<32xi16>, %b: vector<32xi16>, %acc: vector<32xi32>) -> (vector<32xi32>, vector<32xi32>) {
  %ea = arith.extsi %a : vector<32xi16> to vector<32xi32>
  %eb = arith.extsi %b : vector<32xi16> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %r = arith.addi %acc, %p : vector<32xi32>
  return %r, %ea : vector<32xi32>, vector<32xi32>
}

// Standalone widenings.
// CHECK-LABEL: @ext_i8_i16
// CHECK:       aievec.aie4.unpack_i8_i16 %arg0 : vector<32xi8>, vector<32xi16>
// CHECK:       aievec.aie4.unpack_i8_i16 %arg0 {isSigned = false} : vector<32xi8>, vector<32xi16>
// LLVM-LABEL:  @ext_i8_i16
// LLVM:        <{sign = 1 : i32}>
// LLVM:        <{sign = 0 : i32}>
func.func @ext_i8_i16(%a: vector<32xi8>) -> (vector<32xi16>, vector<32xi16>) {
  %s = arith.extsi %a : vector<32xi8> to vector<32xi16>
  %u = arith.extui %a : vector<32xi8> to vector<32xi16>
  return %s, %u : vector<32xi16>, vector<32xi16>
}

// CHECK-LABEL: @ext_i8_i32
// CHECK:       %[[S:.*]] = aievec.aie4.unpack_i8_i16 %arg0 : vector<32xi8>, vector<32xi16>
// CHECK:       aievec.aie4.ups_i16_i32 %[[S]] {shift = 0 : i32} : vector<32xi16>, vector<32xi32>
// CHECK:       %[[U:.*]] = aievec.aie4.unpack_i8_i16 %arg0 {isSigned = false}
// CHECK:       aievec.aie4.ups_i16_i32 %[[U]] {isSigned = false, shift = 0 : i32}
func.func @ext_i8_i32(%a: vector<32xi8>) -> (vector<32xi32>, vector<32xi32>) {
  %s = arith.extsi %a : vector<32xi8> to vector<32xi32>
  %u = arith.extui %a : vector<32xi8> to vector<32xi32>
  return %s, %u : vector<32xi32>, vector<32xi32>
}

// Other shapes are not claimed.
// CHECK-LABEL: @ext_i8_other_shapes
// CHECK:       arith.extsi %arg0 : vector<64xi8> to vector<64xi16>
// CHECK:       arith.extsi %arg1 : vector<16xi8> to vector<16xi16>
func.func @ext_i8_other_shapes(%a: vector<64xi8>, %b: vector<16xi8>) -> (vector<64xi16>, vector<16xi16>) {
  %x = arith.extsi %a : vector<64xi8> to vector<64xi16>
  %y = arith.extsi %b : vector<16xi8> to vector<16xi16>
  return %x, %y : vector<64xi16>, vector<16xi16>
}

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Products used as accumulators, returned, or used outside fusible MACs need
// standalone native lowering. A sum of two products also arises when a
// one-tile GEMV's initial add-to-zero is canonicalized away.
//
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" | FileCheck %s --implicit-check-not=arith.muli --implicit-check-not=arith.extsi --implicit-check-not=arith.extui

// CHECK-LABEL: @two_products
// CHECK: %[[ZERO:.*]] = arith.constant dense<0> : vector<32xi32>
// CHECK: %[[P:.*]] = aievec.aie4.mac_ewise_i16_i32 %{{.*}}, %{{.*}}, %[[ZERO]]
// CHECK: %[[R:.*]] = aievec.aie4.mac_ewise_i16_i32 %{{.*}}, %{{.*}}, %[[P]]
// CHECK: return %[[R]]
func.func @two_products(%a: vector<32xi8>, %b: vector<32xi8>, %c: vector<32xi8>, %d: vector<32xi8>) -> vector<32xi32> {
  %ea = arith.extsi %a : vector<32xi8> to vector<32xi32>
  %eb = arith.extsi %b : vector<32xi8> to vector<32xi32>
  %ec = arith.extsi %c : vector<32xi8> to vector<32xi32>
  %ed = arith.extsi %d : vector<32xi8> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %q = arith.muli %ec, %ed : vector<32xi32>
  %r = arith.addi %p, %q : vector<32xi32>
  return %r : vector<32xi32>
}

// CHECK-LABEL: @shared_product
// CHECK: %[[ZERO:.*]] = arith.constant dense<0> : vector<32xi32>
// CHECK: %[[P:.*]] = aievec.aie4.mac_ewise_i16_i32 %{{.*}}, %{{.*}}, %[[ZERO]]
// CHECK: %[[R:.*]] = aievec.aie4.mac_ewise_i16_i32 %{{.*}}, %{{.*}}, %arg2
// CHECK: return %[[R]], %[[P]]
func.func @shared_product(%a: vector<32xi8>, %b: vector<32xi8>, %acc: vector<32xi32>) -> (vector<32xi32>, vector<32xi32>) {
  %ea = arith.extsi %a : vector<32xi8> to vector<32xi32>
  %eb = arith.extsi %b : vector<32xi8> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %r = arith.addi %acc, %p : vector<32xi32>
  return %r, %p : vector<32xi32>, vector<32xi32>
}

// Both uses belong to a MAC root, but one is its accumulator and survives.
// CHECK-LABEL: @doubled_product
// CHECK: %[[ZERO:.*]] = arith.constant dense<0> : vector<32xi32>
// CHECK: %[[P:.*]] = aievec.aie4.mac_ewise_i16_i32 %{{.*}}, %{{.*}}, %[[ZERO]]
// CHECK: %[[R:.*]] = aievec.aie4.mac_ewise_i16_i32 %{{.*}}, %{{.*}}, %[[P]]
// CHECK: return %[[R]]
func.func @doubled_product(%a: vector<32xi8>, %b: vector<32xi8>) -> vector<32xi32> {
  %ea = arith.extsi %a : vector<32xi8> to vector<32xi32>
  %eb = arith.extsi %b : vector<32xi8> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %r = arith.addi %p, %p : vector<32xi32>
  return %r : vector<32xi32>
}

// Standalone mixed-sign multiplication, with an independent extension use.
// CHECK-LABEL: @mixed_product
// CHECK: %[[ZERO:.*]] = arith.constant dense<0> : vector<32xi32>
// CHECK: %[[A:.*]] = aievec.aie4.unpack_i8_i16 %arg0 {isSigned = false}
// CHECK: %[[EA:.*]] = aievec.aie4.ups_i16_i32 %[[A]] {isSigned = false, shift = 0 : i32}
// CHECK: %[[B:.*]] = aievec.aie4.unpack_i8_i16 %arg1 : vector<32xi8>, vector<32xi16>
// CHECK: %[[P:.*]] = aievec.aie4.mac_ewise_i16_i32 %[[A]], %[[B]], %[[ZERO]]
// CHECK: return %[[P]], %[[EA]]
func.func @mixed_product(%a: vector<32xi8>, %b: vector<32xi8>) -> (vector<32xi32>, vector<32xi32>) {
  %ea = arith.extui %a : vector<32xi8> to vector<32xi32>
  %eb = arith.extsi %b : vector<32xi8> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  return %p, %ea : vector<32xi32>, vector<32xi32>
}

// CHECK-LABEL: @unsigned_product
// CHECK: %[[ZERO:.*]] = arith.constant dense<0> : vector<32xi32>
// CHECK: %[[A:.*]] = aievec.aie4.unpack_i8_i16 %arg0 {isSigned = false}
// CHECK: %[[B:.*]] = aievec.aie4.unpack_i8_i16 %arg1 {isSigned = false}
// CHECK: aievec.aie4.mac_ewise_i16_i32 %[[A]], %[[B]], %[[ZERO]]
func.func @unsigned_product(%a: vector<32xi8>, %b: vector<32xi8>) -> vector<32xi32> {
  %ea = arith.extui %a : vector<32xi8> to vector<32xi32>
  %eb = arith.extui %b : vector<32xi8> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  return %p : vector<32xi32>
}

// i16 operands use the same fallback without unpacking.
// CHECK-LABEL: @i16_product
// CHECK: %[[ZERO:.*]] = arith.constant dense<0> : vector<32xi32>
// CHECK-NOT: unpack
// CHECK: aievec.aie4.mac_ewise_i16_i32 %arg0, %arg1, %[[ZERO]]
func.func @i16_product(%a: vector<32xi16>, %b: vector<32xi16>) -> vector<32xi32> {
  %ea = arith.extsi %a : vector<32xi16> to vector<32xi32>
  %eb = arith.extsi %b : vector<32xi16> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  return %p : vector<32xi32>
}

// A product whose uses all fuse needs no standalone MAC.
// CHECK-LABEL: @two_fused_uses
// CHECK-NOT: arith.constant
// CHECK: %[[R:.*]] = aievec.aie4.mac_ewise_i16_i32 %{{.*}}, %{{.*}}, %arg2
// CHECK: %[[S:.*]] = aievec.aie4.mac_ewise_i16_i32 %{{.*}}, %{{.*}}, %arg3
// CHECK-NOT: aievec.aie4.mac_ewise_i16_i32
// CHECK: return %[[R]], %[[S]]
func.func @two_fused_uses(%a: vector<32xi8>, %b: vector<32xi8>, %acc0: vector<32xi32>, %acc1: vector<32xi32>) -> (vector<32xi32>, vector<32xi32>) {
  %ea = arith.extsi %a : vector<32xi8> to vector<32xi32>
  %eb = arith.extsi %b : vector<32xi8> to vector<32xi32>
  %p = arith.muli %ea, %eb : vector<32xi32>
  %r = arith.addi %acc0, %p : vector<32xi32>
  %s = arith.addi %p, %acc1 : vector<32xi32>
  return %r, %s : vector<32xi32>, vector<32xi32>
}

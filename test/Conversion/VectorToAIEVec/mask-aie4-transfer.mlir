// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --aie4-lower-vector-masks --canonicalize | FileCheck %s

// An inner-inactive lane takes padding even when the outer mask has passthru.
// CHECK-LABEL: @empty_inner_read
// CHECK: %[[PAD:.*]] = arith.constant dense<7> : vector<32xi32>
// CHECK-NOT: memref.load
// CHECK-NOT: vector.load
// CHECK: return %[[PAD]]
func.func @empty_inner_read(%m: memref<64xi32>, %pt: vector<32xi32>) -> vector<32xi32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 7 : i32
  %outer = vector.constant_mask [32] : vector<32xi1>
  %inner = vector.constant_mask [0] : vector<32xi1>
  %r = vector.mask %outer, %pt { vector.transfer_read %m[%c0], %pad, %inner {in_bounds = [true]} : memref<64xi32>, vector<32xi32> } : vector<32xi1> -> vector<32xi32>
  return %r : vector<32xi32>
}

// CHECK-LABEL: @empty_inner_write
// CHECK-NOT: memref.store
// CHECK-NOT: vector.store
// CHECK: return
func.func @empty_inner_write(%m: memref<64xi32>, %v: vector<32xi32>) {
  %c0 = arith.constant 0 : index
  %outer = vector.constant_mask [32] : vector<32xi1>
  %inner = vector.constant_mask [0] : vector<32xi1>
  vector.mask %outer { vector.transfer_write %v, %m[%c0], %inner {in_bounds = [true]} : vector<32xi32>, memref<64xi32> } : vector<32xi1>
  return
}

// Both masks control memory; only the outer mask selects passthru.
// CHECK-LABEL: @arbitrary_masks_read
// CHECK-SAME: %[[OUTER:[a-zA-Z0-9_]+]]: vector<32xi1>, %[[INNER:[a-zA-Z0-9_]+]]: vector<32xi1>, %[[PT:[a-zA-Z0-9_]+]]: vector<32xi32>
// CHECK: %[[OBV:.*]] = vector.bitcast %[[OUTER]]
// CHECK: %[[OB:.*]] = vector.extract %[[OBV]]
// CHECK: %[[IBV:.*]] = vector.bitcast %[[INNER]]
// CHECK: %[[IB:.*]] = vector.extract %[[IBV]]
// CHECK: %[[BOTH:.*]] = arith.andi %[[OB]], %[[IB]] : i32
// CHECK: %[[INIT:.*]] = arith.select %[[OUTER]], %{{.*}}, %[[PT]]
// CHECK: scf.if
// CHECK: vector.load
// CHECK: scf.for
// CHECK: arith.shrui %[[BOTH]],
// CHECK: scf.if
// CHECK: memref.load
func.func @arbitrary_masks_read(%m: memref<64xi32>, %outer: vector<32xi1>, %inner: vector<32xi1>, %pt: vector<32xi32>) -> vector<32xi32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 7 : i32
  %r = vector.mask %outer, %pt { vector.transfer_read %m[%c0], %pad, %inner {in_bounds = [true]} : memref<64xi32>, vector<32xi32> } : vector<32xi1> -> vector<32xi32>
  return %r : vector<32xi32>
}

// Intersecting two prefixes retains a loop without per-lane mask tests.
// CHECK-LABEL: @prefix_masks_write
// CHECK: %[[COUNT:.*]] = arith.minsi %{{.*}}, %{{.*}} : index
// CHECK: %[[COUNT2:.*]] = arith.minsi %{{.*}}, %{{.*}} : index
// CHECK: %[[BOTH:.*]] = arith.minsi %[[COUNT]], %[[COUNT2]] : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[BOTH]]
// CHECK-NOT: scf.if
// CHECK: memref.store
func.func @prefix_masks_write(%m: memref<64xi32>, %i: index, %v: vector<32xi32>, %n: index, %k: index) {
  %outer = vector.create_mask %n : vector<32xi1>
  %inner = vector.create_mask %k : vector<32xi1>
  vector.mask %outer { vector.transfer_write %v, %m[%i], %inner {in_bounds = [true]} : vector<32xi32>, memref<64xi32> } : vector<32xi1>
  return
}

// A negative start skips leading lanes without shifting the values stored.
// CHECK-LABEL: @negative_start_write
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[END:.*]] = arith.constant 4 : index
// CHECK: scf.for %[[LANE:.*]] = %[[ONE]] to %[[END]]
// CHECK: %[[V:.*]] = vector.extract %{{.*}}[%[[LANE]]]
// CHECK: %[[I:.*]] = arith.addi %[[LANE]], %{{.*}} : index
// CHECK: memref.store %[[V]], %{{.*}}[%[[I]]]
func.func @negative_start_write(%m: memref<3xi32>, %v: vector<32xi32>) {
  %neg = arith.constant -1 : index
  %mask = vector.constant_mask [32] : vector<32xi1>
  vector.transfer_write %v, %m[%neg], %mask : vector<32xi32>, memref<3xi32>
  return
}

// Transfers entirely before the memref must return padding without a load.
// CHECK-LABEL: @negative_aligned_read
// CHECK: %[[PAD:.*]] = arith.constant dense<7> : vector<32xi32>
// CHECK-NOT: memref.load
// CHECK-NOT: vector.load
// CHECK: return %[[PAD]]
func.func @negative_aligned_read(%m: memref<64xi32>) -> vector<32xi32> {
  %start = arith.constant -32 : index
  %pad = arith.constant 7 : i32
  %mask = vector.constant_mask [32] : vector<32xi1>
  %r = vector.transfer_read %m[%start], %pad, %mask : memref<64xi32>, vector<32xi32>
  return %r : vector<32xi32>
}

// CHECK-LABEL: @minimum_index_read
// CHECK: %[[PAD:.*]] = arith.constant dense<7> : vector<32xi32>
// CHECK-NOT: memref.load
// CHECK-NOT: vector.load
// CHECK: return %[[PAD]]
func.func @minimum_index_read(%m: memref<64xi32>) -> vector<32xi32> {
  %start = arith.constant -9223372036854775808 : index
  %pad = arith.constant 7 : i32
  %mask = vector.constant_mask [32] : vector<32xi1>
  %r = vector.transfer_read %m[%start], %pad, %mask : memref<64xi32>, vector<32xi32>
  return %r : vector<32xi32>
}

// CHECK-LABEL: @maximum_index_read
// CHECK: %[[PAD:.*]] = arith.constant dense<7> : vector<32xi32>
// CHECK-NOT: memref.load
// CHECK-NOT: vector.load
// CHECK: return %[[PAD]]
func.func @maximum_index_read(%m: memref<64xi32>) -> vector<32xi32> {
  %start = arith.constant 9223372036854775807 : index
  %pad = arith.constant 7 : i32
  %mask = vector.constant_mask [32] : vector<32xi1>
  %r = vector.transfer_read %m[%start], %pad, %mask : memref<64xi32>, vector<32xi32>
  return %r : vector<32xi32>
}

// CHECK-LABEL: @tiny_memref_read
// CHECK: %[[BEGIN:.*]] = arith.constant 5 : index
// CHECK: %[[END:.*]] = arith.constant 8 : index
// CHECK: scf.for %[[LANE:.*]] = %[[BEGIN]] to %[[END]]
// CHECK: %[[I:.*]] = arith.addi %[[LANE]], %{{.*}} : index
// CHECK: %[[V:.*]] = memref.load %{{.*}}[%[[I]]]
// CHECK: vector.insert %[[V]], %{{.*}} [%[[LANE]]]
func.func @tiny_memref_read(%m: memref<3xi32>) -> vector<32xi32> {
  %start = arith.constant -5 : index
  %pad = arith.constant 7 : i32
  %mask = vector.constant_mask [32] : vector<32xi1>
  %r = vector.transfer_read %m[%start], %pad, %mask : memref<3xi32>, vector<32xi32>
  return %r : vector<32xi32>
}

// CHECK-LABEL: @empty_memref_read
// CHECK: %[[PAD:.*]] = arith.constant dense<7> : vector<32xi32>
// CHECK-NOT: memref.load
// CHECK-NOT: vector.load
// CHECK: return %[[PAD]]
func.func @empty_memref_read(%m: memref<0xi32>) -> vector<32xi32> {
  %start = arith.constant -5 : index
  %pad = arith.constant 7 : i32
  %mask = vector.constant_mask [32] : vector<32xi1>
  %r = vector.transfer_read %m[%start], %pad, %mask : memref<0xi32>, vector<32xi32>
  return %r : vector<32xi32>
}
// A prefix combined with an arbitrary mask is packed before intersection.
// CHECK-LABEL: @mixed_masks_read
// CHECK-SAME: %[[INNER:[a-zA-Z0-9_]+]]: vector<32xi1>, %[[PT:[a-zA-Z0-9_]+]]: vector<32xi32>
// CHECK-DAG: %[[PREFIX:.*]] = arith.constant 31 : i32
// CHECK: %[[OUTER:.*]] = vector.constant_mask [5] : vector<32xi1>
// CHECK: %[[IBV:.*]] = vector.bitcast %[[INNER]]
// CHECK: %[[IB:.*]] = vector.extract %[[IBV]]
// CHECK: %[[BOTH:.*]] = arith.andi %[[IB]], %[[PREFIX]] : i32
// CHECK: %[[INIT:.*]] = arith.select %[[OUTER]], %{{.*}}, %[[PT]]
// CHECK: scf.for
// CHECK: arith.shrui %[[BOTH]],
// CHECK: scf.if
// CHECK: memref.load
func.func @mixed_masks_read(%m: memref<64xi32>, %i: index, %inner: vector<32xi1>, %pt: vector<32xi32>) -> vector<32xi32> {
  %pad = arith.constant 7 : i32
  %outer = vector.constant_mask [5] : vector<32xi1>
  %r = vector.mask %outer, %pt { vector.transfer_read %m[%i], %pad, %inner {in_bounds = [true]} : memref<64xi32>, vector<32xi32> } : vector<32xi1> -> vector<32xi32>
  return %r : vector<32xi32>
}

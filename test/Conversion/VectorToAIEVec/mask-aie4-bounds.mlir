// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" | FileCheck %s

// A masked transfer that may run out of bounds also limits its active lanes
// to the memref size. The mask lowering runs before the pipeline's
// canonicalization, which would otherwise mark the transfer in bounds.

// CHECK-LABEL: @read_out_of_bounds
// CHECK-SAME: (%[[M:.*]]: memref<?xf32>, %[[I:.*]]: index, %[[N:.*]]: index)
// CHECK: %[[CNT:.*]] = arith.minsi %{{.*}}, %{{.*}} : index
// CHECK: %[[DIM:.*]] = memref.dim %[[M]], %{{.*}} : memref<?xf32>
// CHECK: %[[NEG:.*]] = arith.maxsi %[[I]], %{{.*}} : index
// CHECK: %[[NEGCLAMP:.*]] = arith.minsi %[[NEG]], %{{.*}} : index
// CHECK: %[[BEGIN:.*]] = arith.subi %{{.*}}, %[[NEGCLAMP]] : index
// CHECK: %[[POS:.*]] = arith.maxsi %[[I]], %{{.*}} : index
// CHECK: %[[REM:.*]] = arith.subi %[[DIM]], %[[POS]] : index
// CHECK: %[[RLO:.*]] = arith.maxsi %[[REM]], %{{.*}} : index
// CHECK: %[[RCNT:.*]] = arith.minsi %[[RLO]], %{{.*}} : index
// CHECK: %[[ENDSUM:.*]] = arith.addi %[[RCNT]], %[[BEGIN]] : index
// CHECK: %[[END:.*]] = arith.minsi %[[ENDSUM]], %{{.*}} : index
// CHECK: %[[LIVE:.*]] = arith.minsi %[[CNT]], %[[END]] : index
// CHECK: scf.for %{{.*}} = %[[BEGIN]] to %[[LIVE]] step
// CHECK: memref.load %[[M]]
func.func @read_out_of_bounds(%m: memref<?xf32>, %i: index, %n: index) -> vector<32xf32> {
  %pad = arith.constant 0.0 : f32
  %mask = vector.create_mask %n : vector<32xi1>
  %v = vector.transfer_read %m[%i], %pad, %mask : memref<?xf32>, vector<32xf32>
  return %v : vector<32xf32>
}

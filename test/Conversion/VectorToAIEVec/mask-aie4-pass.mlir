// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --aie4-lower-vector-masks -split-input-file | FileCheck %s
// RUN: aie-opt %s --aie4-lower-vector-masks="aie-target=aie2" -split-input-file | FileCheck %s --check-prefix=AIE2

// The mask lowering alone, which convert-vector-to-aievec runs first for AIE4
// targets. Other targets are left untouched.
// AIE2-DAG: vector.mask
// AIE2-DAG: vector.create_mask

// Each masked reduction selects its identity into inactive lanes.

// CHECK-LABEL: @id_mul_i32
// CHECK: %[[ID:.*]] = arith.constant dense<1> : vector<32xi32>
// CHECK: %[[S:.*]] = arith.select %{{.*}}, %{{.*}}, %[[ID]] : vector<32xi1>, vector<32xi32>
// CHECK: vector.reduction <mul>
func.func @id_mul_i32(%v: vector<32xi32>, %n: index) -> i32 {
  %mask = vector.create_mask %n : vector<32xi1>
  %r = vector.mask %mask { vector.reduction <mul>, %v : vector<32xi32> into i32 } : vector<32xi1> -> i32
  return %r : i32
}

// CHECK-LABEL: @id_minsi_i32
// CHECK: %[[ID:.*]] = arith.constant dense<2147483647> : vector<32xi32>
// CHECK: %[[S:.*]] = arith.select %{{.*}}, %{{.*}}, %[[ID]] : vector<32xi1>, vector<32xi32>
// CHECK: vector.reduction <minsi>
func.func @id_minsi_i32(%v: vector<32xi32>, %n: index) -> i32 {
  %mask = vector.create_mask %n : vector<32xi1>
  %r = vector.mask %mask { vector.reduction <minsi>, %v : vector<32xi32> into i32 } : vector<32xi1> -> i32
  return %r : i32
}

// CHECK-LABEL: @id_maxsi_i32
// CHECK: %[[ID:.*]] = arith.constant dense<-2147483648> : vector<32xi32>
// CHECK: %[[S:.*]] = arith.select %{{.*}}, %{{.*}}, %[[ID]] : vector<32xi1>, vector<32xi32>
// CHECK: vector.reduction <maxsi>
func.func @id_maxsi_i32(%v: vector<32xi32>, %n: index) -> i32 {
  %mask = vector.create_mask %n : vector<32xi1>
  %r = vector.mask %mask { vector.reduction <maxsi>, %v : vector<32xi32> into i32 } : vector<32xi1> -> i32
  return %r : i32
}

// CHECK-LABEL: @id_minui_i32
// CHECK: %[[ID:.*]] = arith.constant dense<-1> : vector<32xi32>
// CHECK: %[[S:.*]] = arith.select %{{.*}}, %{{.*}}, %[[ID]] : vector<32xi1>, vector<32xi32>
// CHECK: vector.reduction <minui>
func.func @id_minui_i32(%v: vector<32xi32>, %n: index) -> i32 {
  %mask = vector.create_mask %n : vector<32xi1>
  %r = vector.mask %mask { vector.reduction <minui>, %v : vector<32xi32> into i32 } : vector<32xi1> -> i32
  return %r : i32
}

// CHECK-LABEL: @id_maxui_i32
// CHECK: %[[ID:.*]] = arith.constant dense<0> : vector<32xi32>
// CHECK: %[[S:.*]] = arith.select %{{.*}}, %{{.*}}, %[[ID]] : vector<32xi1>, vector<32xi32>
// CHECK: vector.reduction <maxui>
func.func @id_maxui_i32(%v: vector<32xi32>, %n: index) -> i32 {
  %mask = vector.create_mask %n : vector<32xi1>
  %r = vector.mask %mask { vector.reduction <maxui>, %v : vector<32xi32> into i32 } : vector<32xi1> -> i32
  return %r : i32
}

// CHECK-LABEL: @id_and_i32
// CHECK: %[[ID:.*]] = arith.constant dense<-1> : vector<32xi32>
// CHECK: %[[S:.*]] = arith.select %{{.*}}, %{{.*}}, %[[ID]] : vector<32xi1>, vector<32xi32>
// CHECK: vector.reduction <and>
func.func @id_and_i32(%v: vector<32xi32>, %n: index) -> i32 {
  %mask = vector.create_mask %n : vector<32xi1>
  %r = vector.mask %mask { vector.reduction <and>, %v : vector<32xi32> into i32 } : vector<32xi1> -> i32
  return %r : i32
}

// CHECK-LABEL: @id_or_i32
// CHECK: %[[ID:.*]] = arith.constant dense<0> : vector<32xi32>
// CHECK: %[[S:.*]] = arith.select %{{.*}}, %{{.*}}, %[[ID]] : vector<32xi1>, vector<32xi32>
// CHECK: vector.reduction <or>
func.func @id_or_i32(%v: vector<32xi32>, %n: index) -> i32 {
  %mask = vector.create_mask %n : vector<32xi1>
  %r = vector.mask %mask { vector.reduction <or>, %v : vector<32xi32> into i32 } : vector<32xi1> -> i32
  return %r : i32
}

// CHECK-LABEL: @id_xor_i32
// CHECK: %[[ID:.*]] = arith.constant dense<0> : vector<32xi32>
// CHECK: %[[S:.*]] = arith.select %{{.*}}, %{{.*}}, %[[ID]] : vector<32xi1>, vector<32xi32>
// CHECK: vector.reduction <xor>
func.func @id_xor_i32(%v: vector<32xi32>, %n: index) -> i32 {
  %mask = vector.create_mask %n : vector<32xi1>
  %r = vector.mask %mask { vector.reduction <xor>, %v : vector<32xi32> into i32 } : vector<32xi1> -> i32
  return %r : i32
}

// CHECK-LABEL: @id_mul_f32
// CHECK: %[[ID:.*]] = arith.constant dense<1.000000e+00> : vector<32xf32>
// CHECK: %[[IDI:.*]] = vector.bitcast %[[ID]] : vector<32xf32> to vector<32xi32>
// CHECK: %[[S:.*]] = arith.select %{{.*}}, %{{.*}}, %[[IDI]] : vector<32xi1>, vector<32xi32>
// CHECK: vector.reduction <mul>
func.func @id_mul_f32(%v: vector<32xf32>, %n: index) -> f32 {
  %mask = vector.create_mask %n : vector<32xi1>
  %r = vector.mask %mask { vector.reduction <mul>, %v : vector<32xf32> into f32 } : vector<32xi1> -> f32
  return %r : f32
}

// CHECK-LABEL: @id_minimumf_f32
// CHECK: %[[ID:.*]] = arith.constant dense<0x7F800000> : vector<32xf32>
// CHECK: %[[IDI:.*]] = vector.bitcast %[[ID]] : vector<32xf32> to vector<32xi32>
// CHECK: %[[S:.*]] = arith.select %{{.*}}, %{{.*}}, %[[IDI]] : vector<32xi1>, vector<32xi32>
// CHECK: vector.reduction <minimumf>
func.func @id_minimumf_f32(%v: vector<32xf32>, %n: index) -> f32 {
  %mask = vector.create_mask %n : vector<32xi1>
  %r = vector.mask %mask { vector.reduction <minimumf>, %v : vector<32xf32> into f32 } : vector<32xi1> -> f32
  return %r : f32
}

// -----

// Active lanes past the memref end take the padding; masked-off lanes take
// the passthru. Only lanes that are both active and in bounds are loaded.
// CHECK-LABEL: @read_out_of_bounds_passthru
// CHECK-SAME: (%[[M:.*]]: memref<?xf32>, %[[I:.*]]: index, %[[N:.*]]: index, %[[PT:.*]]: vector<32xf32>)
// CHECK: %[[MASK:.*]] = vector.bitcast %{{.*}} : vector<1xi32> to vector<32xi1>
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
// CHECK: %[[PAD:.*]] = vector.broadcast %{{.*}} : f32 to vector<32xf32>
// CHECK: %[[PADI:.*]] = vector.bitcast %[[PAD]] : vector<32xf32> to vector<32xi32>
// CHECK: %[[PTI:.*]] = vector.bitcast %[[PT]] : vector<32xf32> to vector<32xi32>
// CHECK: %[[INITI:.*]] = arith.select %[[MASK]], %[[PADI]], %[[PTI]] : vector<32xi1>, vector<32xi32>
// CHECK: %[[INIT:.*]] = vector.bitcast %[[INITI]] : vector<32xi32> to vector<32xf32>
// CHECK: scf.for %{{.*}} = %[[BEGIN]] to %[[LIVE]] step %{{.*}} iter_args(%{{.*}} = %[[INIT]])
// CHECK: memref.load %[[M]]
func.func @read_out_of_bounds_passthru(%m: memref<?xf32>, %i: index, %n: index, %pt: vector<32xf32>) -> vector<32xf32> {
  %pad = arith.constant 7.0 : f32
  %mask = vector.create_mask %n : vector<32xi1>
  %v = vector.mask %mask, %pt { vector.transfer_read %m[%i], %pad : memref<?xf32>, vector<32xf32> } : vector<32xi1> -> vector<32xf32>
  return %v : vector<32xf32>
}

// -----

// The lane count is clamped in index arithmetic before it is narrowed:
// 2^32 + 2 lanes selects all 32 lanes, not 2.
// CHECK-LABEL: @create_mask_huge_count
// CHECK: %[[N:.*]] = arith.constant 4294967298 : index
// CHECK: %[[LO:.*]] = arith.maxsi %[[N]], %{{.*}} : index
// CHECK: %[[CNT:.*]] = arith.minsi %[[LO]], %{{.*}} : index
// CHECK: arith.index_cast %[[CNT]] : index to i32
func.func @create_mask_huge_count(%a: vector<32xi32>, %b: vector<32xi32>) -> vector<32xi32> {
  %n = arith.constant 4294967298 : index
  %mask = vector.create_mask %n : vector<32xi1>
  %r = arith.select %mask, %a, %b : vector<32xi1>, vector<32xi32>
  return %r : vector<32xi32>
}

// -----

// Runtime sequences are host code and are left alone.
// CHECK-LABEL: aie.runtime_sequence
// CHECK: vector.maskedstore
module {
  aie.device(npu3) {
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %c0 = arith.constant 0 : index
      %mask = vector.constant_mask [5] : vector<32xi1>
      %v = arith.constant dense<1> : vector<32xi32>
      vector.maskedstore %arg0[%c0], %mask, %v : memref<64xi32>, vector<32xi1>, vector<32xi32>
    }
  }
}

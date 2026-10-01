// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" -split-input-file | FileCheck %s

// Use vector accesses only for fully active, provably aligned accesses;
// otherwise access active lanes individually. Inactive lanes retain padding
// or passthru values.

// The form linalg vectorization emits: an aligned loop with a create_mask of
// the remaining lanes. Full chunks use vector accesses; the tail loops over
// exactly the active lanes, with no per-lane test.
// CHECK-LABEL: @loop_create_mask
// CHECK-SAME: (%[[A:.*]]: memref<1000xf32>, %[[C:.*]]: memref<1000xf32>)
// CHECK-DAG: %[[PAD:.*]] = arith.constant dense<7.000000e+00> : vector<32xf32>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK: scf.for %[[I:.*]] = %[[C0]] to %{{.*}} step %[[C32]] {
// CHECK:   %[[N:.*]] = arith.minsi %{{.*}}, %[[C32]] : index
// CHECK:   %[[LO:.*]] = arith.maxsi %[[N]], %[[C0]] : index
// CHECK:   %[[CNT:.*]] = arith.minsi %[[LO]], %[[C32]] : index
// CHECK:   %[[FULL:.*]] = arith.cmpi eq, %[[CNT]], %[[C32]] : index
// CHECK:   scf.if %[[FULL]] {
// CHECK:     %[[V:.*]] = vector.load %[[A]][%[[I]]] : memref<1000xf32>, vector<32xf32>
// CHECK:     vector.store %[[V]], %[[C]][%[[I]]] : memref<1000xf32>, vector<32xf32>
// CHECK:   } else {
// CHECK:     %[[T:.*]] = scf.for %[[J:.*]] = %[[C0]] to %[[CNT]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[PAD]]) -> (vector<32xf32>) {
// CHECK-NEXT:  %[[IJ:.*]] = arith.addi %[[I]], %[[J]] : index
// CHECK-NEXT:  %[[X:.*]] = memref.load %[[A]][%[[IJ]]] : memref<1000xf32>
// CHECK-NEXT:  %[[INS:.*]] = vector.insert %[[X]], %[[ACC]] [%[[J]]] : f32 into vector<32xf32>
// CHECK-NEXT:  scf.yield %[[INS]]
// CHECK:     scf.for %[[K:.*]] = %[[C0]] to %[[CNT]] step %[[C1]] {
// CHECK-NEXT:  %[[Y:.*]] = vector.extract %[[T]][%[[K]]] : f32 from vector<32xf32>
// CHECK-NEXT:  %[[IK:.*]] = arith.addi %[[I]], %[[K]] : index
// CHECK-NEXT:  memref.store %[[Y]], %[[C]][%[[IK]]] : memref<1000xf32>
// CHECK-NOT: vector.mask
func.func @loop_create_mask(%a: memref<1000xf32>, %c: memref<1000xf32>) {
  %pad = arith.constant 7.0 : f32
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1000 = arith.constant 1000 : index
  scf.for %i = %c0 to %c1000 step %c32 {
    %r = arith.subi %c1000, %i : index
    %n = arith.minsi %r, %c32 : index
    %mask = vector.create_mask %n : vector<32xi1>
    %v = vector.mask %mask { vector.transfer_read %a[%i], %pad {in_bounds = [true]} : memref<1000xf32>, vector<32xf32> } : vector<32xi1> -> vector<32xf32>
    vector.mask %mask { vector.transfer_write %v, %c[%i] {in_bounds = [true]} : vector<32xf32>, memref<1000xf32> } : vector<32xi1>
  }
  return
}

// -----

// A full mask at an unaligned index must not use a whole-vector access: VLD
// and VST would drop the 7 low address bits and move c[0..31] instead.
// CHECK-LABEL: @full_mask_unaligned
// CHECK-NOT: vector.load
// CHECK-NOT: vector.store
// CHECK: scf.for
// CHECK: memref.load
// CHECK: scf.for
// CHECK: memref.store
// CHECK: return
func.func @full_mask_unaligned(%a: memref<64xi32>, %c: memref<64xi32>) {
  %c3 = arith.constant 3 : index
  %m = vector.constant_mask [32] : vector<32xi1>
  %pt = arith.constant dense<0> : vector<32xi32>
  %v = vector.maskedload %a[%c3], %m, %pt : memref<64xi32>, vector<32xi1>, vector<32xi32> into vector<32xi32>
  vector.maskedstore %c[%c3], %m, %v : memref<64xi32>, vector<32xi1>, vector<32xi32>
  return
}

// Unknown alignment (a function argument index) also takes the scalar path.
// CHECK-LABEL: @write_unknown_alignment
// CHECK-NOT: vector.store
// CHECK: memref.store
func.func @write_unknown_alignment(%m: memref<1000xf32>, %i: index, %n: index, %v: vector<32xf32>) {
  %mask = vector.create_mask %n : vector<32xi1>
  vector.mask %mask { vector.transfer_write %v, %m[%i] {in_bounds = [true]} : vector<32xf32>, memref<1000xf32> } : vector<32xi1>
  return
}

// A subview whose static offset is not a multiple of 32 is not aligned.
// CHECK-LABEL: @write_unaligned_subview
// CHECK-NOT: vector.store
// CHECK: memref.store
func.func @write_unaligned_subview(%m: memref<1088xf32>, %n: index, %v: vector<32xf32>) {
  %c0 = arith.constant 0 : index
  %s = memref.subview %m[35] [1000] [1] : memref<1088xf32> to memref<1000xf32, strided<[1], offset: 35>>
  %mask = vector.create_mask %n : vector<32xi1>
  vector.mask %mask { vector.transfer_write %v, %s[%c0] {in_bounds = [true]} : vector<32xf32>, memref<1000xf32, strided<[1], offset: 35>> } : vector<32xi1>
  return
}

// An aligned subview offset keeps the vector path.
// CHECK-LABEL: @write_aligned_subview
// CHECK: vector.store
func.func @write_aligned_subview(%m: memref<1088xf32>, %n: index, %v: vector<32xf32>) {
  %c64 = arith.constant 64 : index
  %s = memref.subview %m[32] [1000] [1] : memref<1088xf32> to memref<1000xf32, strided<[1], offset: 32>>
  %mask = vector.create_mask %n : vector<32xi1>
  vector.mask %mask { vector.transfer_write %v, %s[%c64] {in_bounds = [true]} : vector<32xf32>, memref<1000xf32, strided<[1], offset: 32>> } : vector<32xi1>
  return
}

// -----

// A vector.mask passthru replaces the padding for inactive lanes.
// CHECK-LABEL: @read_passthru
// CHECK-SAME: %[[PT:[[:alnum:]]+]]: vector<32xi32>)
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[PT]]) -> (vector<32xi32>)
// CHECK: memref.load
func.func @read_passthru(%m: memref<1000xi32>, %i: index, %n: index, %pt: vector<32xi32>) -> vector<32xi32> {
  %pad = arith.constant 0 : i32
  %mask = vector.create_mask %n : vector<32xi1>
  %v = vector.mask %mask, %pt { vector.transfer_read %m[%i], %pad {in_bounds = [true]} : memref<1000xi32>, vector<32xi32> } : vector<32xi1> -> vector<32xi32>
  return %v : vector<32xi32>
}

// -----

// A general mask (here from a native compare) is packed into bits and each
// lane is tested; an aligned constant index keeps the full-mask vector path.
// CHECK-LABEL: @maskedload_cmp
// CHECK-SAME: %[[PT:[[:alnum:]]+]]: vector<32xi32>)
// CHECK: %[[CMP:.*]] = aievec.aie4.cmp
// CHECK: %[[BITS:.*]] = builtin.unrealized_conversion_cast %[[CMP]] : ui32 to i32
// CHECK: %[[FULL:.*]] = arith.cmpi eq, %[[BITS]], %{{.*}} : i32
// CHECK: scf.if %[[FULL]]
// CHECK:   vector.load
// CHECK: } else {
// CHECK:   scf.for %[[J:.*]] = {{.*}} iter_args(%{{.*}} = %[[PT]])
// CHECK:     %[[SJ:.*]] = arith.index_cast %[[J]] : index to i32
// CHECK:     %[[SH:.*]] = arith.shrui %[[BITS]], %[[SJ]] : i32
// CHECK:     %[[B:.*]] = arith.andi %[[SH]], %{{.*}} : i32
// CHECK:     %[[ON:.*]] = arith.trunci %[[B]] : i32 to i1
// CHECK:     scf.if %[[ON]]
// CHECK:       memref.load
func.func @maskedload_cmp(%m: memref<64xi32>, %a: vector<32xi32>, %b: vector<32xi32>, %pt: vector<32xi32>) -> vector<32xi32> {
  %c32 = arith.constant 32 : index
  %mask = arith.cmpi slt, %a, %b : vector<32xi32>
  %v = vector.maskedload %m[%c32], %mask, %pt : memref<64xi32>, vector<32xi1>, vector<32xi32> into vector<32xi32>
  return %v : vector<32xi32>
}

// -----

// A constant partial mask stores exactly its 5 lanes.
// CHECK-LABEL: @maskedstore_constant_mask
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : index
// CHECK-NOT: vector.store
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[C5]] step
// CHECK: memref.store
func.func @maskedstore_constant_mask(%m: memref<64xf32>, %v: vector<32xf32>) {
  %c0 = arith.constant 0 : index
  %mask = vector.constant_mask [5] : vector<32xi1>
  vector.maskedstore %m[%c0], %mask, %v : memref<64xf32>, vector<32xi1>, vector<32xf32>
  return
}

// -----

// 64-lane i16 vectors: full 64-lane chunks store the whole vector.
// CHECK-LABEL: @write_i16_64
// CHECK: %[[CNT:.*]] = arith.minsi %{{.*}}, %[[C64:.*]] : index
// CHECK: arith.cmpi eq, %[[CNT]], %[[C64]] : index
// CHECK: vector.store %{{.*}} : memref<128xi16>, vector<64xi16>
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[CNT]] step
// CHECK: memref.store %{{.*}} : memref<128xi16>
func.func @write_i16_64(%m: memref<128xi16>, %n: index, %v: vector<64xi16>) {
  %c64 = arith.constant 64 : index
  %mask = vector.create_mask %n : vector<64xi1>
  vector.mask %mask { vector.transfer_write %v, %m[%c64] {in_bounds = [true]} : vector<64xi16>, memref<128xi16> } : vector<64xi1>
  return
}

// -----

// Masked reductions select the identity into inactive lanes with VSEL.
// CHECK-LABEL: @reduce_add_i32
// CHECK-SAME: (%[[V:.*]]: vector<32xi32>, %[[N:.*]]: index, %[[ACC:.*]]: i32)
// CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0> : vector<32xi32>
// CHECK: %[[BITS:.*]] = arith.select %{{.*}} : i32
// CHECK: %[[SEL:.*]] = builtin.unrealized_conversion_cast %[[BITS]] : i32 to ui32
// CHECK: %[[S:.*]] = aievec.aie4.sel %[[ZERO]], %[[V]], %[[SEL]]
// CHECK: vector.reduction <add>, %[[S]], %[[ACC]] : vector<32xi32> into i32
func.func @reduce_add_i32(%v: vector<32xi32>, %n: index, %acc: i32) -> i32 {
  %mask = vector.create_mask %n : vector<32xi1>
  %r = vector.mask %mask { vector.reduction <add>, %v, %acc : vector<32xi32> into i32 } : vector<32xi1> -> i32
  return %r : i32
}

// CHECK-LABEL: @reduce_maximumf_f32
// CHECK-SAME: (%[[V:.*]]: vector<32xf32>
// CHECK-DAG: %[[NINF:.*]] = arith.constant dense<0xFF800000> : vector<32xf32>
// CHECK-DAG: %[[VI:.*]] = vector.bitcast %[[V]] : vector<32xf32> to vector<32xi32>
// CHECK-DAG: %[[NI:.*]] = vector.bitcast %[[NINF]] : vector<32xf32> to vector<32xi32>
// CHECK: %[[S:.*]] = aievec.aie4.sel %[[NI]], %[[VI]], %{{.*}}
// CHECK: %[[F:.*]] = vector.bitcast %[[S]] : vector<32xi32> to vector<32xf32>
// CHECK: aievec.aie4.reduce <max> %[[F]] : vector<32xf32> -> f32
func.func @reduce_maximumf_f32(%v: vector<32xf32>, %n: index) -> f32 {
  %mask = vector.create_mask %n : vector<32xi1>
  %r = vector.mask %mask { vector.reduction <maximumf>, %v : vector<32xf32> into f32 } : vector<32xi1> -> f32
  return %r : f32
}

// Linalg vectorization masks 1-D multi_reduction; -0.0 is the add identity.
// CHECK-LABEL: @multi_reduction_add_f32
// CHECK-DAG: %[[NZ:.*]] = arith.constant dense<-0.000000e+00> : vector<32xf32>
// CHECK: vector.bitcast %[[NZ]]
// CHECK: aievec.aie4.sel
// CHECK: vector.reduction <add>, %{{.*}}, %{{.*}} : vector<32xf32> into f32
func.func @multi_reduction_add_f32(%v: vector<32xf32>, %n: index, %acc: f32) -> f32 {
  %mask = vector.create_mask %n : vector<32xi1>
  %r = vector.mask %mask { vector.multi_reduction <add>, %v, %acc [0] : vector<32xf32> to f32 } : vector<32xi1> -> f32
  return %r : f32
}

// -----

// A create_mask that feeds a select becomes scalar bits and VSEL.
// CHECK-LABEL: @select_create_mask
// CHECK-SAME: (%[[A:.*]]: vector<32xi32>, %[[B:.*]]: vector<32xi32>, %[[N:.*]]: index)
// CHECK-NOT: vector.create_mask
// CHECK: %[[BITS:.*]] = arith.select %{{.*}} : i32
// CHECK: %[[SEL:.*]] = builtin.unrealized_conversion_cast %[[BITS]] : i32 to ui32
// CHECK: aievec.aie4.sel %[[B]], %[[A]], %[[SEL]]
func.func @select_create_mask(%a: vector<32xi32>, %b: vector<32xi32>, %n: index) -> vector<32xi32> {
  %mask = vector.create_mask %n : vector<32xi1>
  %r = arith.select %mask, %a, %b : vector<32xi1>, vector<32xi32>
  return %r : vector<32xi32>
}

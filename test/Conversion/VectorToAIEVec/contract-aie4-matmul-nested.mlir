// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --lower-vector-to-aievec="aie-target=aie4b" | FileCheck %s
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" | FileCheck %s

// Matmul stages its own B tile, so a contract inside an active matrix region
// must remain generic even when its types and indexing maps otherwise match.
// The contract deliberately uses a different B tile than the enclosing region.
// CHECK-LABEL: func.func @directly_in_matrix_region(
// CHECK: aievec.aie4.matrix_region int8
// CHECK-NOT: aievec.aie4.matmul
// CHECK: %[[R:.*]] = vector.contract
// CHECK-NOT: aievec.aie4.matmul
// CHECK: aievec.aie4.matrix_yield %[[R]]
// CHECK-NOT: aievec.aie4.matmul
// CHECK: return
func.func @directly_in_matrix_region(%a: vector<1x64xi8>, %b: vector<64x64xi8>, %b2: vector<64x64xi8>, %c: vector<1x64xi32>) -> vector<1x64xi32> {
  %r = aievec.aie4.matrix_region int8 %b : vector<64x64xi8> -> vector<1x64xi32> attributes {rhs_signed = true} {
    %aw = arith.extsi %a : vector<1x64xi8> to vector<1x64xi32>
    %bw = arith.extsi %b2 : vector<64x64xi8> to vector<64x64xi32>
    %v = vector.contract {indexing_maps = [
                            affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                          iterator_types = ["parallel", "parallel", "reduction"],
                          kind = #vector.kind<add>}
         %aw, %bw, %c : vector<1x64xi32>, vector<64x64xi32> into vector<1x64xi32>
    aievec.aie4.matrix_yield %v : vector<1x64xi32>
  }
  return %r : vector<1x64xi32>
}

// The matrix-region restriction applies through nested control flow too.
// CHECK-LABEL: func.func @under_if_in_matrix_region(
// CHECK: aievec.aie4.matrix_region int8
// CHECK: scf.if
// CHECK-NOT: aievec.aie4.matmul
// CHECK: %[[R:.*]] = vector.contract
// CHECK-NOT: aievec.aie4.matmul
// CHECK: scf.yield %[[R]]
// CHECK-NOT: aievec.aie4.matmul
// CHECK: aievec.aie4.matrix_yield
// CHECK-NOT: aievec.aie4.matmul
// CHECK: return
func.func @under_if_in_matrix_region(%cond: i1, %a: vector<1x64xi8>, %b: vector<64x64xi8>, %b2: vector<64x64xi8>, %c: vector<1x64xi32>) -> vector<1x64xi32> {
  %r = aievec.aie4.matrix_region int8 %b : vector<64x64xi8> -> vector<1x64xi32> attributes {rhs_signed = true} {
    %v = scf.if %cond -> (vector<1x64xi32>) {
      %aw = arith.extsi %a : vector<1x64xi8> to vector<1x64xi32>
      %bw = arith.extsi %b2 : vector<64x64xi8> to vector<64x64xi32>
      %contract = vector.contract {indexing_maps = [
                                     affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                                   iterator_types = ["parallel", "parallel", "reduction"],
                                   kind = #vector.kind<add>}
                  %aw, %bw, %c : vector<1x64xi32>, vector<64x64xi32> into vector<1x64xi32>
      scf.yield %contract : vector<1x64xi32>
    } else {
      scf.yield %c : vector<1x64xi32>
    }
    aievec.aie4.matrix_yield %v : vector<1x64xi32>
  }
  return %r : vector<1x64xi32>
}

// Control flow alone does not prevent native matmul conversion.
// CHECK-LABEL: func.func @under_if_outside_matrix_region(
// CHECK: scf.if
// CHECK-NOT: vector.contract
// CHECK: %[[R:.*]] = aievec.aie4.matmul
// CHECK-NOT: vector.contract
// CHECK: scf.yield %[[R]]
// CHECK-NOT: vector.contract
// CHECK: return
func.func @under_if_outside_matrix_region(%cond: i1, %a: vector<1x64xi8>, %b: vector<64x64xi8>, %c: vector<1x64xi32>) -> vector<1x64xi32> {
  %r = scf.if %cond -> (vector<1x64xi32>) {
    %aw = arith.extsi %a : vector<1x64xi8> to vector<1x64xi32>
    %bw = arith.extsi %b : vector<64x64xi8> to vector<64x64xi32>
    %v = vector.contract {indexing_maps = [
                            affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                          iterator_types = ["parallel", "parallel", "reduction"],
                          kind = #vector.kind<add>}
         %aw, %bw, %c : vector<1x64xi32>, vector<64x64xi32> into vector<1x64xi32>
    scf.yield %v : vector<1x64xi32>
  } else {
    scf.yield %c : vector<1x64xi32>
  }
  return %r : vector<1x64xi32>
}

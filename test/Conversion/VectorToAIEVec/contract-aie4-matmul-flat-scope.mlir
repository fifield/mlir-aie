// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file --lower-vector-to-aievec="aie-target=aie4b" | FileCheck %s
// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie4b" | FileCheck %s

// The creation guard does not affect contractions in execution scopes
// without flat matrix operations. Rejection cases, including live primed
// and committed tiles, are in contract-aie4-matmul-flat-scope-refuse.mlir.

// Without flat ops the contraction still becomes a matmul.
// CHECK-LABEL: func.func @no_flat_ops
// CHECK: aievec.aie4.matmul
// CHECK-NOT: aievec.aie4.matrix_region
func.func @no_flat_ops(%a: vector<2x64xi8>, %b: vector<64x64xi8>, %c: vector<2x64xi32>) -> vector<2x64xi32> {
  %aw = arith.extsi %a : vector<2x64xi8> to vector<2x64xi32>
  %bw = arith.extsi %b : vector<64x64xi8> to vector<64x64xi32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<2x64xi32>, vector<64x64xi32> into vector<2x64xi32>
  return %r : vector<2x64xi32>
}

// -----

// The guard looks at the contraction's own scope: flat ops in another
// function do not affect it.
// CHECK-LABEL: func.func @flat_elsewhere
// CHECK: aievec.aie4.stage
// CHECK-LABEL: func.func @contract_here
// CHECK: aievec.aie4.matmul
// CHECK-NOT: aievec.aie4.matrix_region
func.func @flat_elsewhere(%chunk: vector<128xi8>, %s: i32) {
  aievec.aie4.stage int8 %chunk, %s {rhs_signed = true} : vector<128xi8>, i32
  return
}
func.func @contract_here(%a: vector<2x64xi8>, %b: vector<64x64xi8>, %c: vector<2x64xi32>) -> vector<2x64xi32> {
  %aw = arith.extsi %a : vector<2x64xi8> to vector<2x64xi32>
  %bw = arith.extsi %b : vector<64x64xi8> to vector<64x64xi32>
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<2x64xi32>, vector<64x64xi32> into vector<2x64xi32>
  return %r : vector<2x64xi32>
}

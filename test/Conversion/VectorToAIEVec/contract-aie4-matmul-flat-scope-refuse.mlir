// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file --aievec-aie4-check-staging -o /dev/null
// RUN: not aie-opt %s -split-input-file --lower-vector-to-aievec="aie-target=aie4b" 2>&1 | FileCheck %s
// RUN: not aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie4b" 2>&1 | FileCheck %s

// A partial prime is enough to prohibit native contraction lowering in the
// execution scope, even without a later commit. Ordering a generated region
// does not preserve the user's staging contents.
func.func @partial_prime(%chunk: vector<128xi8>, %slot: i32, %a: vector<2x64xi8>, %b: vector<64x64xi8>, %c: vector<2x64xi32>) -> vector<2x64xi32> {
  aievec.aie4.stage int8 %chunk, %slot {rhs_signed = true} : vector<128xi8>, i32
  %aw = arith.extsi %a : vector<2x64xi8> to vector<2x64xi32>
  %bw = arith.extsi %b : vector<64x64xi8> to vector<64x64xi32>
  // CHECK: warning: AIE4: native vector.contract lowering must not share an execution scope with flat staging operations
  // CHECK: error: failed to legalize operation 'vector.contract'
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<2x64xi32>, vector<64x64xi32> into vector<2x64xi32>
  return %r : vector<2x64xi32>
}

// -----

// Fully prime B0, compute a pure contraction with B1, then commit B0 and use
// it. Previously lowering inserted B1's 32 stages and commit before the
// user's commit, silently replacing B0; the staging lint accepted both tiles
// because it checks format provenance rather than tile identity.
func.func @primed_tile(%chunk: vector<128xi8>, %a: vector<2x64xi8>, %b: vector<64x64xi8>, %c: vector<2x64xi32>) -> (vector<2x64xi32>, vector<64xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c32 = arith.constant 32 : i32
  scf.for %slot = %c0 to %c32 step %c1 : i32 {
    aievec.aie4.stage int8 %chunk, %slot {rhs_signed = true} : vector<128xi8>, i32
  }
  %aw = arith.extsi %a : vector<2x64xi8> to vector<2x64xi32>
  %bw = arith.extsi %b : vector<64x64xi8> to vector<64x64xi32>
  // CHECK: warning: {{.*}}staging and committing its B tile could replace a primed or committed tile
  // CHECK: error: failed to legalize operation 'vector.contract'
  %r = vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<2x64xi32>, vector<64x64xi32> into vector<2x64xi32>
  aievec.aie4.commit int8 {rhs_signed = true}
  %row = vector.extract %a[0] : vector<64xi8> from vector<2x64xi8>
  %flat = aievec.aie4.mmac_active int8 %row {lhs_signed = true, rhs_signed = true} : vector<64xi8> -> vector<64xi32>
  return %r, %flat : vector<2x64xi32>, vector<64xi32>
}

// -----

// A committed B0 is also live across a contraction using B1. Cover stream
// staging, a floating native format, and a contraction nested in control flow:
// the enclosing execution scope's flat ops must still prevent the lowering.
func.func @committed_tile(%port: i32, %cond: i1, %a: vector<2x32xbf16>, %b: vector<32x64xbf16>, %c: vector<2x64xf32>) -> (vector<2x64xf32>, vector<64xf32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c32 = arith.constant 32 : i32
  scf.for %slot = %c0 to %c32 step %c1 : i32 {
    aievec.aie4.stage_stream bf16 %port, %slot : i32, i32
  }
  aievec.aie4.commit bf16
  %r = scf.if %cond -> (vector<2x64xf32>) {
    %aw = arith.extf %a : vector<2x32xbf16> to vector<2x32xf32>
    %bw = arith.extf %b : vector<32x64xbf16> to vector<32x64xf32>
    // CHECK: warning: {{.*}}Use explicit matrix operations where replacing the tile is safe
    // CHECK: error: failed to legalize operation 'vector.contract'
    %contract = vector.contract {indexing_maps = [
                                  affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                                iterator_types = ["parallel", "parallel", "reduction"],
                                kind = #vector.kind<add>, fastmath = #arith.fastmath<afn>}
         %aw, %bw, %c : vector<2x32xf32>, vector<32x64xf32> into vector<2x64xf32>
    scf.yield %contract : vector<2x64xf32>
  } else {
    scf.yield %c : vector<2x64xf32>
  }
  %row = vector.extract %a[0] : vector<32xbf16> from vector<2x32xbf16>
  %flat = aievec.aie4.mmac_active bf16 %row : vector<32xbf16> -> vector<64xf32>
  return %r, %flat : vector<2x64xf32>, vector<64xf32>
}

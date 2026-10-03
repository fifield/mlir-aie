// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Mask lowering precedes MMAC matching in both entry points and rejects
// the rank-3 contraction mask. Check the specific diagnostic rather than
// accepting an arbitrary failure or a retained contraction.
//
// RUN: aie-opt %s --lower-vector-to-aievec="aie-target=aie4b" -verify-diagnostics
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" -verify-diagnostics

func.func @masked_rejected(%a: vector<4x64xi8>, %b: vector<64x64xi8>, %c: vector<4x64xi32>, %m: vector<4x64x64xi1>) -> vector<4x64xi32> {
  %aw = arith.extsi %a : vector<4x64xi8> to vector<4x64xi32>
  %bw = arith.extsi %b : vector<64x64xi8> to vector<64x64xi32>
  // expected-error @+1 {{AIE4: masks must be fixed 1-D with at most 64 lanes}}
  %r = vector.mask %m { vector.contract {indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
       %aw, %bw, %c : vector<4x64xi32>, vector<64x64xi32> into vector<4x64xi32> } : vector<4x64x64xi1> -> vector<4x64xi32>
  return %r : vector<4x64xi32>
}

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --lower-vector-to-aievec="aie-target=aie4b" | FileCheck %s
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" | FileCheck %s

// Unsupported vector shapes must remain on the generic arithmetic path.
// CHECK-LABEL: func.func @mac_srs_scalable
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: arith.shrsi
// CHECK: arith.trunci
// CHECK-NOT: aievec.aie4.
func.func @mac_srs_scalable(%a: vector<[32]xi16>, %b: vector<[32]xi16>,
                                 %acc: vector<[32]xi32>) -> vector<[32]xi16> {
  %ea = arith.extsi %a : vector<[32]xi16> to vector<[32]xi32>
  %eb = arith.extsi %b : vector<[32]xi16> to vector<[32]xi32>
  %p = arith.muli %ea, %eb : vector<[32]xi32>
  %m = arith.addi %acc, %p : vector<[32]xi32>
  %sh = arith.constant dense<4> : vector<[32]xi32>
  %s = arith.shrsi %m, %sh : vector<[32]xi32>
  %r = arith.trunci %s : vector<[32]xi32> to vector<[32]xi16>
  return %r : vector<[32]xi16>
}

// -----

// Unsupported vector shapes must remain on the generic arithmetic path.
// CHECK-LABEL: func.func @mac_srs_rank_two
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: arith.shrsi
// CHECK: arith.trunci
// CHECK-NOT: aievec.aie4.
func.func @mac_srs_rank_two(%a: vector<4x8xi16>, %b: vector<4x8xi16>,
                                 %acc: vector<4x8xi32>) -> vector<4x8xi16> {
  %ea = arith.extsi %a : vector<4x8xi16> to vector<4x8xi32>
  %eb = arith.extsi %b : vector<4x8xi16> to vector<4x8xi32>
  %p = arith.muli %ea, %eb : vector<4x8xi32>
  %m = arith.addi %acc, %p : vector<4x8xi32>
  %sh = arith.constant dense<4> : vector<4x8xi32>
  %s = arith.shrsi %m, %sh : vector<4x8xi32>
  %r = arith.trunci %s : vector<4x8xi32> to vector<4x8xi16>
  return %r : vector<4x8xi16>
}

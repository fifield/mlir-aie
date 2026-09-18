//===- target-parsing-aievec-to-llvm.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pins that convert-aievec-to-llvm's switch from AIEVecTargetInfo is still
// the deliberately-binary isAIE2P() ? AIE2p : AIE2 -- not a three-way arch
// switch -- and did not regress when the pass was made to go through
// xilinx::aievec::parseAIEVecTarget instead of comparing the raw
// "aie-target" string by hand. "aieml" is just an alias for "aie2" here, the
// same as in the other four AIEVec sub-pipelines (see target-parsing.mlir).
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie2" | FileCheck %s --check-prefix=AIE2
// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aieml" | FileCheck %s --check-prefix=AIE2
// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s --check-prefix=AIE2P

// AIE2: xllvm.intr.aie2.add.accfloat
// AIE2-NOT: vector.shuffle
// AIE2P: xllvm.intr.aie2p.ACC2048.accfloat.add.conf
func.func @add_elem_flat_fp32(%lhs : vector<16xf32>, %rhs : vector<16xf32>) -> vector<16xf32> {
  %0 = aievec.add_elem %lhs, %rhs : vector<16xf32>
  return %0 : vector<16xf32>
}

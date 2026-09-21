//
// target-parsing-aievec-to-llvm-aie4.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// There is exactly one AIE4 core variant in practice (AIE4b/medusa): `aie4a`
// is descoped, and `--mcpu` has been measured to be inert for AIEVec
// lowering. So a bare "aie4" is accepted, exactly like "aie4b" (see
// target-parsing-aie4.mlir for the parity pin across all four AIEVec
// sub-pipelines). This matters concretely because `detectAIETarget`
// (tools/aiecc/IRTransforms.h) can only ever emit the bare "aie4" spelling.
//
// AIE4's AIEVec-to-LLVM arm covers only aievec.aie4.mul_ewise_f32 and does
// not fall through to the common/AIE2 patterns, so an AIE2 op
// (`aievec.add_elem`) must fail to legalize under the bare "aie4" spelling
// exactly as it does under "aie4b" (see
// target-parsing-aievec-to-llvm-aie4b-with-op.mlir).
//
// RUN: not aie-opt %s --convert-aievec-to-llvm="aie-target=aie4" 2>&1 | FileCheck %s
// CHECK: failed to legalize operation 'aievec.add_elem'

func.func @add_elem_flat_fp32(%lhs : vector<16xf32>, %rhs : vector<16xf32>) -> vector<16xf32> {
  %0 = aievec.add_elem %lhs, %rhs : vector<16xf32>
  return %0 : vector<16xf32>
}

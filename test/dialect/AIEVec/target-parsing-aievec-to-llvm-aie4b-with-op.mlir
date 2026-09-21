//
// target-parsing-aievec-to-llvm-aie4b-with-op.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AIE4 now has an AIEVec-to-LLVM lowering, but it covers exactly one op
// (aievec.aie4.mul_ewise_f32). `populateAIEVecToLLVMConversionPatterns` gives
// AIE4 its own arm that adds *only* the AIE4 patterns -- it does not fall
// through to the common/AIE2 patterns.
//
// This pins that anti-fallthrough property: `aievec.add_elem` is an AIE2 op
// with a perfectly good AIE2 lowering, so if AIE4 ever started sharing AIE2's
// pattern set this would silently start succeeding. Under an AIE4 target it
// must instead fail to legalize, with a non-zero exit code.
//
// Previously this produced an "AIE4 AIEVec lowering is not enabled"
// diagnostic (the whole target was disabled); now the target is enabled and
// the failure is a per-op legalization failure, which is a strictly narrower
// and stronger guarantee.
//
// RUN: not aie-opt %s --convert-aievec-to-llvm="aie-target=aie4b" 2>&1 | FileCheck %s
// CHECK: failed to legalize operation 'aievec.add_elem'

func.func @add_elem_flat_fp32(%lhs : vector<16xf32>, %rhs : vector<16xf32>) -> vector<16xf32> {
  %0 = aievec.add_elem %lhs, %rhs : vector<16xf32>
  return %0 : vector<16xf32>
}

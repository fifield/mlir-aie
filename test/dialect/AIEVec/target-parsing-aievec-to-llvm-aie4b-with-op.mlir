// expected-error@-1 {{AIEVec lowering is not enabled for target aie4b}}
//
// target-parsing-aievec-to-llvm-aie4b-with-op.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// "aie4b" is a validly-parsed target (unlike "aie4", "aie4a", "aie2ps"), but
// there is no AIEVec-to-LLVM lowering for AIE4 yet
// (isAIEVecToLLVMLoweringEnabled returns false for AIE4). When the module
// being converted actually contains an AIEVec op, convert-aievec-to-llvm
// must fail with a "not enabled" diagnostic and a non-zero exit code, rather
// than silently falling through to the AIE2 patterns as it did before this
// fix. Contrast with target-parsing-aievec-to-llvm-aie4b-no-op.mlir, which
// pins the opposite case (no AIEVec ops -> pass cleanly), since today's AIE4
// aiecc flows never contain AIEVec ops and an unconditional failure here
// would turn every NPU3 build in programming_examples/aie4/ red. See
// target-parsing-unknown.mlir for why `expected-error@-1` on line 1 matches
// the error attached to the sentinel (file:0:0) module location.
//
// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie4b" -verify-diagnostics

func.func @add_elem_flat_fp32(%lhs : vector<16xf32>, %rhs : vector<16xf32>) -> vector<16xf32> {
  %0 = aievec.add_elem %lhs, %rhs : vector<16xf32>
  return %0 : vector<16xf32>
}

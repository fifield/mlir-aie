// expected-error@-1 {{AIEVec lowering is not enabled for target aie4b}}
//
// target-parsing-aievec-to-llvm-aie4.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// There is exactly one AIE4 core variant in practice (AIE4b/medusa): `aie4a`
// is descoped, and `--mcpu` has been measured to be inert for AIEVec
// lowering. So a bare "aie4" is accepted, exactly like "aie4b" (see
// target-parsing-aie4.mlir for the parity pin across all five AIEVec
// sub-pipelines). There is still no AIEVec-to-LLVM lowering for AIE4, so
// when the module actually contains an AIEVec op, convert-aievec-to-llvm
// must fail with the same "not enabled" diagnostic as "aie4b" does (see
// target-parsing-aievec-to-llvm-aie4b-with-op.mlir). Contrast with
// target-parsing-aievec-to-llvm-aie4-no-op.mlir, which pins the opposite
// case (no AIEVec ops -> pass cleanly, exit 0) -- this is what keeps
// today's AIE4 aiecc flows green, since `detectAIETarget`
// (tools/aiecc/IRTransforms.h) can only ever emit bare "aie4". See
// target-parsing-unknown.mlir for why `expected-error@-1` on line 1 matches
// the error attached to the sentinel (file:0:0) module location.
//
// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie4" -verify-diagnostics

func.func @add_elem_flat_fp32(%lhs : vector<16xf32>, %rhs : vector<16xf32>) -> vector<16xf32> {
  %0 = aievec.add_elem %lhs, %rhs : vector<16xf32>
  return %0 : vector<16xf32>
}

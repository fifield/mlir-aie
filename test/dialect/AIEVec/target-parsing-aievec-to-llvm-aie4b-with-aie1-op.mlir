// expected-error@-1 {{AIEVec lowering is not enabled for target aie4b}}
//
// target-parsing-aievec-to-llvm-aie4b-with-aie1-op.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The op-presence walk in ConvertAIEVecToLLVMPass checks
// `isa<AIEVecDialect, aie1::AIEVecAIE1Dialect>(dialect)` -- two dialects,
// not one. Every other test in this family (aie4b-with-op,
// aie4-with-op via target-parsing-aievec-to-llvm-aie4.mlir) only ever uses
// an `aievec.*` op, so it only ever exercises the `AIEVecDialect` arm; an
// `aievec_aie1.*` op could be silently dropped from the `isa<>` and no
// existing test would notice. This test pins the `aie1::AIEVecAIE1Dialect`
// arm specifically, using an AIE1 op (`aievec_aie1.mac`) with an AIE4
// target: it must still be treated as "an AIEVec op is present" and fail
// with the same "not enabled" diagnostic. See
// target-parsing-unknown.mlir for why `expected-error@-1` on line 1 matches
// the error attached to the sentinel (file:0:0) module location.
//
// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie4b" -verify-diagnostics

func.func @mac_f32(%A : vector<16xf32>, %B : vector<8xf32>, %ACC : vector<8xf32>) -> vector<8xf32> {
  %0 = aievec_aie1.mac %A, %B, %ACC {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
  return %0 : vector<8xf32>
}

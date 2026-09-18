// expected-error@-1 {{AIE2PS is not supported}}
//
// target-parsing-aie2ps.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// "aie2ps" (VE3858-class devices) has no AIEVec vector lowering and must be
// rejected by all four AIEVec sub-pipelines as an unsupported target. This
// covers only the aie-opt side; the aiecc/AIETargetModel half of aie2ps
// handling is out of scope for this PR. See target-parsing-unknown.mlir for why
// `expected-error@-1` on line 1 matches the error attached to the sentinel
// (file:0:0) module location.
//
// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie2ps" -verify-diagnostics
// RUN: aie-opt %s --lower-vector-to-aievec="aie-target=aie2ps" -verify-diagnostics
// RUN: aie-opt %s --optimize-aievec="aie-target=aie2ps" -verify-diagnostics
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2ps" -verify-diagnostics

func.func @f() {
  return
}

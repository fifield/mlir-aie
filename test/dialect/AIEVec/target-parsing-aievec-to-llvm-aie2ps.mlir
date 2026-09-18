// expected-error@-1 {{AIE2PS is not a supported AIEVec target}}
//
// target-parsing-aievec-to-llvm-aie2ps.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// "aie2ps" (VE3858-class devices) has no AIEVec vector lowering and must be
// rejected by convert-aievec-to-llvm with a message explaining why (as
// opposed to the generic "unknown AIE target" message), same as the other
// four AIEVec sub-pipelines. See target-parsing-unknown.mlir for why
// `expected-error@-1` on line 1 matches the error attached to the sentinel
// (file:0:0) module location.
//
// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie2ps" -verify-diagnostics

func.func @f() {
  return
}

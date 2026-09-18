// expected-error@-1 {{unknown AIE target 'notatarget'}}
//
// target-parsing-aievec-to-llvm-unknown.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// convert-aievec-to-llvm is the fifth AIEVec pipeline consumer of
// xilinx::aievec::parseAIEVecTarget, and until now it validated nothing:
// populateAIEVecToLLVMConversionPatterns did a bare
// `aieTarget == "aie2p" ? AIE2p : AIE2`, so an unrecognized `aie-target`
// spelling silently performed a full AIE2 lowering and exited 0. This test
// pins the fix: an unknown target must be rejected, byte-identically with
// the other four sub-pipelines. See target-parsing-unknown.mlir for why
// `expected-error@-1` on line 1 matches the error attached to the sentinel
// (file:0:0) module location.
//
// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=notatarget" -verify-diagnostics

func.func @f() {
  return
}

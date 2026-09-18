// expected-error@-1 {{unknown AIE target 'notatarget'}}
//
// target-parsing-unknown.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// An unrecognized `aie-target` spelling must be rejected, byte-identically,
// by all four AIEVec sub-pipelines. `-verify-diagnostics` works here even
// though the error is attached to the top-level module op, whose location
// is a sentinel (file:0:0): that sentinel resolves to line 0, and
// `expected-error@-1` placed on line 1 (above) resolves to line 0 too, so
// it matches.
//
// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=notatarget" -verify-diagnostics
// RUN: aie-opt %s --lower-vector-to-aievec="aie-target=notatarget" -verify-diagnostics
// RUN: aie-opt %s --optimize-aievec="aie-target=notatarget" -verify-diagnostics
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=notatarget" -verify-diagnostics

func.func @f() {
  return
}

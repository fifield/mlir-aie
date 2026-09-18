// expected-error@-1 {{Vector canonicalization is not enabled for target aie4b}}
//
// target-parsing-aie4b-canon.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// There are no AIEVec optimization patterns for AIE4 yet
// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie4b" -verify-diagnostics
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" -verify-diagnostics

func.func @f() {
  return
}

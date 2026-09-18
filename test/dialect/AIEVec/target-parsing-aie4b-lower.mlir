// expected-error@-1 {{Vector lowering is not enabled for target aie4b}}
//
// target-parsing-aie4b-lower.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// There are no AIEVec optimization patterns for AIE4 yet
// RUN: aie-opt %s --lower-vector-to-aievec="aie-target=aie4b" -verify-diagnostics

func.func @f() {
  return
}

// expected-error@-1 {{AIEVec has no vector lowering for target 'aie2ps'}}
//
// target-parsing-aie2ps.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// "aie2ps" has no AIEVec vector lowering
//
// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie2ps" -verify-diagnostics
// RUN: aie-opt %s --lower-vector-to-aievec="aie-target=aie2ps" -verify-diagnostics
// RUN: aie-opt %s --optimize-aievec="aie-target=aie2ps" -verify-diagnostics
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2ps" -verify-diagnostics

func.func @f() {
  return
}

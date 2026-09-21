//
// target-parsing-aie4b-canon.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Test that the pass *runs* and exits 0 on an AIE4 target;
//
// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie4b" -verify-diagnostics | FileCheck %s
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @f
func.func @f() {
  return
}

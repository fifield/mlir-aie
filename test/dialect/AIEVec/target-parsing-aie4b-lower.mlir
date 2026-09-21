//
// target-parsing-aie4b-lower.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt %s --lower-vector-to-aievec="aie-target=aie4b" -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @f
func.func @f() {
  return
}

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A known target with no AIEVec ops requires no lowering.
// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie2ps" | FileCheck %s
// CHECK-LABEL: func.func @f()
// CHECK-NEXT: return
func.func @f() {
  return
}

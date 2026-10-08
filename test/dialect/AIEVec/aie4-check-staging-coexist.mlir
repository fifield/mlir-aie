// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: not aie-opt %s -split-input-file -aievec-aie4-check-staging="phase=coexistence" 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: aie-opt %s -split-input-file -aievec-aie4-check-staging="phase=coexistence strict=false" 2>&1 | FileCheck %s --check-prefix=WARN --implicit-check-not=error:

// A matmul must not share an execution scope with flat ops.

// WARN: warning: 'aievec.aie4.matmul' must not share an execution scope with flat staging operations
// WARN-NOT: error:
func.func @matmul_with_stage(%chunk: vector<64xbf16>, %s: i32, %a: vector<2x32xbf16>, %b: vector<32x64xbf16>, %c: vector<2x64xf32>) -> vector<2x64xf32> {
  aievec.aie4.stage bf16 %chunk, %s : vector<64xbf16>, i32
  // ERROR: error: 'aievec.aie4.matmul' must not share an execution scope with flat staging operations; a moved matmul could replace a primed or committed tile. Use 'aievec.aie4.matrix_region' here
  // ERROR: note: flat staging operation here
  %r = aievec.aie4.matmul %a, %b, %c : vector<2x32xbf16>, vector<32x64xbf16> into vector<2x64xf32>
  return %r : vector<2x64xf32>
}

// -----

// Different cores are different scopes.
aie.device(npu3) {
  %t0 = aie.tile(0, 2)
  %t1 = aie.tile(0, 3)
  %core0 = aie.core(%t0) {
    %a = arith.constant dense<0.0> : vector<2x32xbf16>
    %b = arith.constant dense<0.0> : vector<32x64xbf16>
    %c = arith.constant dense<0.0> : vector<2x64xf32>
    %r = aievec.aie4.matmul %a, %b, %c : vector<2x32xbf16>, vector<32x64xbf16> into vector<2x64xf32>
    aie.end
  }
  %core1 = aie.core(%t1) {
    %p = arith.constant 0 : i32
    aievec.aie4.stage_stream bf16 %p, %p : i32, i32
    aie.end
  }
}

// -----

// A helper function called from a core is its own scope.
aie.device(npu3) {
  func.func @helper(%a: vector<2x32xbf16>, %b: vector<32x64xbf16>, %c: vector<2x64xf32>) -> vector<2x64xf32> {
    %r = aievec.aie4.matmul %a, %b, %c : vector<2x32xbf16>, vector<32x64xbf16> into vector<2x64xf32>
    return %r : vector<2x64xf32>
  }
  %t0 = aie.tile(0, 2)
  %core0 = aie.core(%t0) {
    %p = arith.constant 0 : i32
    %a = arith.constant dense<0.0> : vector<2x32xbf16>
    %b = arith.constant dense<0.0> : vector<32x64xbf16>
    %c = arith.constant dense<0.0> : vector<2x64xf32>
    aievec.aie4.stage_stream bf16 %p, %p : i32, i32
    %r = func.call @helper(%a, %b, %c) : (vector<2x32xbf16>, vector<32x64xbf16>, vector<2x64xf32>) -> vector<2x64xf32>
    aie.end
  }
}

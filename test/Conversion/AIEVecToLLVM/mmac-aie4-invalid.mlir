// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -split-input-file -aievec-aie4-expand-matmul -verify-diagnostics
// RUN: aie-opt %s -split-input-file -convert-aievec-to-llvm="aie-target=aie4b" -verify-diagnostics

// The lowering keeps one accumulator register per row live, so it accepts at
// most 64 rows (kAIE4MatMulMaxRows).
func.func @too_many_rows(%a: vector<65x32xbf16>, %b: vector<32x64xbf16>, %c: vector<65x64xf32>) -> vector<65x64xf32> {
  // expected-error @+1 {{has 65 rows; lowering supports at most 64 (one live accumulator register per row); tile M}}
  %0 = aievec.aie4.matmul %a, %b, %c : vector<65x32xbf16>, vector<32x64xbf16> into vector<65x64xf32>
  return %0 : vector<65x64xf32>
}

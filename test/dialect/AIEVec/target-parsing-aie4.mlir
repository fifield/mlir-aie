//===- target-parsing-aie4.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie4" > %t.aie4-canon.out
// RUN: aie-opt %s --canonicalize-vector-for-aievec="aie-target=aie4b" > %t.aie4b-canon.out
// RUN: diff %t.aie4-canon.out %t.aie4b-canon.out

// RUN: aie-opt %s --lower-vector-to-aievec="aie-target=aie4" > %t.aie4-lower.out
// RUN: aie-opt %s --lower-vector-to-aievec="aie-target=aie4b" > %t.aie4b-lower.out
// RUN: diff %t.aie4-lower.out %t.aie4b-lower.out

// RUN: aie-opt %s --optimize-aievec="aie-target=aie4" > %t.aie4-opt.out
// RUN: aie-opt %s --optimize-aievec="aie-target=aie4b" > %t.aie4b-opt.out
// RUN: diff %t.aie4-opt.out %t.aie4b-opt.out

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4" > %t.aie4-conv.out
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie4b" > %t.aie4b-conv.out
// RUN: diff %t.aie4-conv.out %t.aie4b-conv.out

// A vector<32xf32> multiply, so the two spellings are compared on a body that
// actually exercises the AIE4 lowering rather than on an empty function.
func.func @f(%a : vector<32xf32>, %b : vector<32xf32>) -> vector<32xf32> {
  %0 = arith.mulf %a, %b : vector<32xf32>
  return %0 : vector<32xf32>
}

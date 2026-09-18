//===- target-parsing-aievec-to-llvm-aie4b-no-op.mlir -----------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// "aie4b" is a validly-parsed target with no AIEVec-to-LLVM lowering yet,
// but today's AIE4 aiecc flows never emit AIEVec ops in the first place.
// convert-aievec-to-llvm must pass cleanly (exit 0, no diagnostic) on a
// module that targets AIE4 and contains no AIEVec/AIEVecAIE1 ops -- this is
// what keeps every NPU3 build in programming_examples/aie4/ green. Contrast
// with target-parsing-aievec-to-llvm-aie4b-with-op.mlir, which pins the
// opposite case (an AIEVec op present -> hard failure).
//
// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie4b"

func.func @f(%a : f32, %b : f32) -> f32 {
  %0 = arith.addf %a, %b : f32
  return %0 : f32
}

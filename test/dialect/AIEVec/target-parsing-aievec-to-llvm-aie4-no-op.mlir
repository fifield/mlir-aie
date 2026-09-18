//===- target-parsing-aievec-to-llvm-aie4-no-op.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A bare "aie4" is accepted (there is exactly one AIE4 core variant in
// practice, so "aie4" and "aie4b" parse identically -- see
// target-parsing-aie4.mlir), but there is still no AIEVec-to-LLVM lowering
// for AIE4 (isAIEVecToLLVMLoweringEnabled returns false). The pass only
// fails if the module actually contains an AIEVec op to lower; today's AIE4
// aiecc flows never emit AIEVec ops in the first place, so
// convert-aievec-to-llvm must pass cleanly (exit 0, no diagnostic) on a
// module that targets bare "aie4" and contains no AIEVec/AIEVecAIE1 ops --
// this is what keeps every NPU3 build in programming_examples/aie4/ green.
// Contrast with target-parsing-aievec-to-llvm-aie4.mlir, which pins the
// opposite case (an AIEVec op present -> hard failure).
//
// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie4"

func.func @f(%a : f32, %b : f32) -> f32 {
  %0 = arith.addf %a, %b : f32
  return %0 : f32
}

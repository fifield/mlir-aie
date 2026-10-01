//
// aie4-nlf-effects.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INV and INVSQRT read crFPNlfMask and write the srFPNlf status flags, so
// identical calls are not merged and unused ones are not removed.
//
// RUN: aie-opt %s -split-input-file -cse | FileCheck %s --check-prefix=CSE
// RUN: aie-opt %s -split-input-file -canonicalize | FileCheck %s --check-prefix=DCE

// CSE-LABEL: llvm.func @inv_pair_not_merged
// CSE: %[[A:.*]] = "xllvm.intr.aie4.inv"
// CSE: %[[B:.*]] = "xllvm.intr.aie4.inv"
// CSE: llvm.fadd %[[A]], %[[B]]
llvm.func @inv_pair_not_merged(%x: f32) -> f32 {
  %0 = "xllvm.intr.aie4.inv"(%x) : (f32) -> f32
  %1 = "xllvm.intr.aie4.inv"(%x) : (f32) -> f32
  %2 = llvm.fadd %0, %1 : f32
  llvm.return %2 : f32
}

// -----

// CSE-LABEL: llvm.func @invsqrt_pair_not_merged
// CSE: %[[A:.*]] = "xllvm.intr.aie4.invsqrt"
// CSE: %[[B:.*]] = "xllvm.intr.aie4.invsqrt"
// CSE: llvm.fadd %[[A]], %[[B]]
llvm.func @invsqrt_pair_not_merged(%x: f32) -> f32 {
  %0 = "xllvm.intr.aie4.invsqrt"(%x) : (f32) -> f32
  %1 = "xllvm.intr.aie4.invsqrt"(%x) : (f32) -> f32
  %2 = llvm.fadd %0, %1 : f32
  llvm.return %2 : f32
}

// -----

// DCE-LABEL: llvm.func @unused_kept
// DCE: "xllvm.intr.aie4.inv"
// DCE: "xllvm.intr.aie4.invsqrt"
llvm.func @unused_kept(%x: f32) {
  %0 = "xllvm.intr.aie4.inv"(%x) : (f32) -> f32
  %1 = "xllvm.intr.aie4.invsqrt"(%x) : (f32) -> f32
  llvm.return
}

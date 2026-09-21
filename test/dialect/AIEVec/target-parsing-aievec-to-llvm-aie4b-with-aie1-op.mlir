//
// target-parsing-aievec-to-llvm-aie4b-with-aie1-op.mlir
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Companion to target-parsing-aievec-to-llvm-aie4b-with-op.mlir, covering the
// *other* AIEVec dialect. That test uses an `aievec.*` op; this one uses an
// `aievec_aie1.*` op (`aievec_aie1.mac`), whose lowering lives in the common
// pattern set that AIE4 deliberately does not pull in.
//
// Under an AIE4 target it must fail to legalize rather than be lowered by
// AIE1's patterns.
//
// RUN: not aie-opt %s --convert-aievec-to-llvm="aie-target=aie4b" 2>&1 | FileCheck %s
// CHECK: failed to legalize operation 'aievec_aie1.mac'

func.func @mac_f32(%A : vector<16xf32>, %B : vector<8xf32>, %ACC : vector<8xf32>) -> vector<8xf32> {
  %0 = aievec_aie1.mac %A, %B, %ACC {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
  return %0 : vector<8xf32>
}

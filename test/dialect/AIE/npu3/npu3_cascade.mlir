//===- npu3_cascade.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// AIE4 cascades are 512 bits wide and, as on AIE2, run north to south and
// west to east.

// RUN: aie-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: aie.device(npu3)
// CHECK: aie.configure_cascade(%{{.*}}, North, East)
// CHECK: aie.get_cascade() : vector<16xi32>
// CHECK: aie.put_cascade(%{{.*}} : vector<16xi32>)
aie.device(npu3) {
  %t13 = aie.tile(1, 3)
  aie.configure_cascade(%t13, North, East)
  %c13 = aie.core(%t13) {
    %v = aie.get_cascade() : vector<16xi32>
    aie.put_cascade(%v : vector<16xi32>)
    aie.end
  }
}

// -----

aie.device(npu3) {
  %t13 = aie.tile(1, 3)
  // expected-error@+1 {{input direction of cascade must be North or West on AIE4}}
  aie.configure_cascade(%t13, East, South)
}

// -----

aie.device(npu3) {
  %t13 = aie.tile(1, 3)
  // expected-error@+1 {{output direction of cascade must be South or East on AIE4}}
  aie.configure_cascade(%t13, North, West)
}

// -----

aie.device(npu3) {
  %t13 = aie.tile(1, 3)
  %c13 = aie.core(%t13) {
    // expected-error@+1 {{must be a 512-bit type}}
    %v = aie.get_cascade() : vector<8xi32>
    aie.end
  }
}

// -----

aie.device(npu3) {
  %t13 = aie.tile(1, 3)
  %c13 = aie.core(%t13) {
    %v = arith.constant dense<0> : vector<8xi32>
    // expected-error@+1 {{type must match architecture cascade width (512 bits in AIE4)}}
    aie.put_cascade(%v : vector<8xi32>)
    aie.end
  }
}

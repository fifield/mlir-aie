//===- exclusive_crossbar_outputs_overflow.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s

// Four flows cross from row 3 to row 2, which has four South channels, and a
// fifth ends at (0, 3) South:0. A circuit takes its output alone, so the
// router fails rather than driving South:0 twice.

// CHECK-NOT: error: 'aie.connect'
// CHECK:     error: Unable to find a legal routing: the router found no routing that avoids two streams driving (0, 3) South:0, which a circuit stream takes alone.

module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.flow(%t03, DMA : 0, %t03, South : 0)
    aie.flow(%t04, DMA : 0, %t01, DMA : 0)
    aie.flow(%t04, DMA : 1, %t01, DMA : 1)
    aie.flow(%t05, DMA : 0, %t01, DMA : 2)
    aie.flow(%t05, DMA : 1, %t01, DMA : 3)
  }
}

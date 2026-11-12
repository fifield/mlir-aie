//===- exclusive_crossbar_outputs.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-split-flow-vias --aie-create-pathfinder-flows %s | FileCheck %s

// A flow that ends at a switchbox port drives that crossbar output but takes
// no wire beyond it, so only the crossbar sees the output in use. A circuit
// takes its output alone: no other stream may be routed onto it. Here three
// flows cross from row 3 to row 2, which has four South channels, one of them
// the output a flow ends at.

// A circuit ends at (0, 3) South:0.
// CHECK-LABEL: aie.device(npu1_1col)
// CHECK:       aie.switchbox(%{{.*}}tile_0_3) {
// CHECK-COUNT-1: South : 0>
// CHECK-NOT:     South : 0>
// CHECK:       }

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
  }
}

// -----

// A via splits a flow into flows that end and start at switchbox ports; the
// first ends at (0, 3) South:1.
// CHECK-LABEL: aie.device(npu1_1col)
// CHECK:       aie.switchbox(%{{.*}}tile_0_3) {
// CHECK-COUNT-1: South : 1>
// CHECK-NOT:     South : 1>
// CHECK:       }

module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.flow(%t04, DMA : 1, %t02, DMA : 1) via (%t04 : DMA : 1 -> South : 1, %t03 : North : 1 -> South : 1, %t02 : North : 1 -> DMA : 1)
    aie.flow(%t04, DMA : 0, %t02, DMA : 0)
    aie.flow(%t05, DMA : 0, %t01, DMA : 0)
    aie.flow(%t05, DMA : 1, %t01, DMA : 1)
  }
}

// -----

// A packet flow ends at (0, 3) South:0, so a circuit may not drive it.
// CHECK-LABEL: aie.device(npu1_1col)
// CHECK:       aie.switchbox(%{{.*}}tile_0_3) {
// CHECK:         aie.masterset(South : 0
// CHECK-NOT:     South : 0>
// CHECK:       }

module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.packet_flow(1) {
      aie.packet_source<%t03, DMA : 0>
      aie.packet_dest<%t03, South : 0>
    }
    aie.flow(%t04, DMA : 0, %t01, DMA : 0)
    aie.flow(%t04, DMA : 1, %t01, DMA : 1)
    aie.flow(%t05, DMA : 0, %t01, DMA : 2)
  }
}

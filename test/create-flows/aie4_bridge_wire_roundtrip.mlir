//===- aie4_bridge_wire_roundtrip.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows %s | FileCheck %s --check-prefix=WIRE
// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s --check-prefix=FLOW

// Recover a generated circuit route that currently crosses the bridge twice.
// The flow check also remains valid if routing costs later favor a wide path.
// FLOW-LABEL: aie.device(npu3_1col)
// FLOW: %[[MEM:.*]] = aie.tile(0, 1)
// FLOW: aie.flow(%[[MEM]], DMA : 0, %[[MEM]], DMA : 2)
// WIRE-LABEL: aie.device(npu3_1col)

module {
  aie.device(npu3_1col) {
    %mem = aie.tile(0, 1)
    aie.flow(%mem, DMA : 0, %mem, DMA : 2)
  }
}

// -----

// An explicit circuit tests both bridge directions independently of route
// selection. Channel 1 leaves the 512-bit switch; channel 0 returns to it.
// WIRE-LABEL: aie.device(npu3_1col)
// WIRE: %[[SW:.*]] = aie.switchbox
// WIRE: aie.wire(%[[SW]] : Switch512, %[[SW]] : Switch32)
// FLOW-LABEL: aie.device(npu3_1col)
// FLOW: %[[MEM:.*]] = aie.tile(0, 1)
// FLOW: aie.flow(%[[MEM]], DMA : 0, %[[MEM]], DMA : 2)
// FLOW-NOT: aie.flow

module {
  aie.device(npu3_1col) {
    %mem = aie.tile(0, 1)
    aie.switchbox(%mem) {
      aie.connect<DMA : 0, Switch32 : 1>
      aie.connect<Switch512 : 1, Switch512 : 0>
      aie.connect<Switch32 : 0, DMA : 2>
    }
  }
}

// -----

// Recover a packet route through the same pair of bridge wires.
// WIRE-LABEL: aie.device(npu3_1col)
// WIRE: %[[SW:.*]] = aie.switchbox
// WIRE: aie.wire(%[[SW]] : Switch512, %[[SW]] : Switch32)
// FLOW-LABEL: aie.device(npu3_1col)
// FLOW: %[[MEM:.*]] = aie.tile(0, 1)
// FLOW: aie.packet_flow(7) {
// FLOW-NEXT: aie.packet_source<%[[MEM]], DMA : 0>
// FLOW-NEXT: aie.packet_dest<%[[MEM]], DMA : 2>
// FLOW-NEXT: }
// FLOW-NOT: aie.packet_flow

module {
  aie.device(npu3_1col) {
    %mem = aie.tile(0, 1)
    aie.switchbox(%mem) {
      %to32 = aie.amsel<0> (0)
      %to512 = aie.amsel<1> (0)
      %toDMA = aie.amsel<2> (0)
      aie.masterset(Switch32 : 1, %to32)
      aie.masterset(Switch512 : 0, %to512)
      aie.masterset(DMA : 2, %toDMA)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 7, %to32)
      }
      aie.packet_rules(Switch512 : 1) {
        aie.rule(31, 7, %to512)
      }
      aie.packet_rules(Switch32 : 0) {
        aie.rule(31, 7, %toDMA)
      }
    }
  }
}

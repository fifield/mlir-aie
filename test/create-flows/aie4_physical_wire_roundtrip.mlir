// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows %s > %t.once
// RUN: FileCheck %s --check-prefix=WIRE < %t.once
// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows %t.once > %t.twice
// RUN: diff %t.once %t.twice
// RUN: aie-opt --split-input-file --aie-find-flows %t.once > %t.lifted
// RUN: FileCheck %s --check-prefix=FLOW < %t.lifted
// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows --aie-find-flows %t.lifted | FileCheck %s --check-prefix=FLOW

// AIE4 data and dedicated trace DMA attach directly to the shim switch.
// High source channels exceed each tile's receive-channel count and must
// still be recovered. Both endpoints must survive lifting, with one copy
// of each wire. Bridge-only recovery is covered by aie4_bridge_wire_roundtrip.
// WIRE-LABEL: aie.device(npu3_1col)
// WIRE-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
// WIRE-DAG: %[[MEM:.*]] = aie.tile(0, 1)
// WIRE: %[[SS:.*]] = aie.switchbox(%[[SHIM]])
// WIRE: %[[MS:.*]] = aie.switchbox(%[[MEM]])
// WIRE-DAG: aie.wire(%[[SHIM]] : DMA, %[[SS]] : DMA)
// WIRE-DAG: aie.wire(%[[SHIM]] : DMATrace, %[[SS]] : DMATrace)
// WIRE-DAG: aie.wire(%[[SS]] : North, %[[MS]] : South)
// FLOW-LABEL: aie.device(npu3_1col)
// FLOW-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
// FLOW-DAG: %[[MEM:.*]] = aie.tile(0, 1)
// FLOW-DAG: aie.flow(%[[SHIM]], DMA : 3, %[[MEM]], DMA : 0)
// FLOW-DAG: aie.flow(%[[MEM]], DMA : 9, %[[SHIM]], DMATrace : 0)
// FLOW-NOT: aie.flow
module {
  aie.device(npu3_1col) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    aie.flow(%shim, DMA : 3, %mem, DMA : 0)
    aie.flow(%mem, DMA : 9, %shim, DMATrace : 0)
  }
}

// -----

// Same-tile trace circuits are real endpoint flows. The non-Core/DMA source
// must not be treated as an orphan fabric entry and dropped during lifting.
// Explicit connects require a narrow-to-wide bridge independently of routing
// preferences; the shared RUN lines also reroute and lift the recovered flow.
// WIRE-LABEL: aie.device(npu3_1col)
// WIRE: %[[SHIM:.*]] = aie.tile(0, 0)
// WIRE: %[[SS:.*]] = aie.switchbox(%[[SHIM]])
// WIRE-DAG: aie.wire(%[[SHIM]] : Trace32, %[[SS]] : Trace32)
// WIRE-DAG: aie.wire(%[[SHIM]] : DMATrace, %[[SS]] : DMATrace)
// WIRE-DAG: aie.wire(%[[SS]] : Switch512, %[[SS]] : Switch32)
// FLOW-LABEL: aie.device(npu3_1col)
// FLOW: %[[SHIM:.*]] = aie.tile(0, 0)
// FLOW: aie.flow(%[[SHIM]], Trace32 : 0, %[[SHIM]], DMATrace : 0)
// FLOW-NOT: aie.flow
module {
  aie.device(npu3_1col) {
    %shim = aie.tile(0, 0)
    aie.switchbox(%shim) {
      aie.connect<Trace32 : 0, Switch512 : 0>
      aie.connect<Switch32 : 0, DMATrace : 0>
    }
  }
}

// -----

// Force both vertical and horizontal narrow routes. Without physical wires,
// the lifter mistakes narrow ports for endpoint tiles and loses the true sink.
// WIRE-LABEL: aie.device(npu3)
// WIRE-DAG: %[[SHIM0:.*]] = aie.tile(0, 0)
// WIRE-DAG: %[[SHIM1:.*]] = aie.tile(1, 0)
// WIRE-DAG: %[[MEM:.*]] = aie.tile(0, 1)
// WIRE-DAG: %[[CORE:.*]] = aie.tile(0, 2)
// WIRE-DAG: %[[S0:.*]] = aie.switchbox(%[[SHIM0]])
// WIRE-DAG: %[[S1:.*]] = aie.switchbox(%[[SHIM1]])
// WIRE-DAG: %[[MS:.*]] = aie.switchbox(%[[MEM]])
// WIRE-DAG: %[[CS:.*]] = aie.switchbox(%[[CORE]])
// WIRE-DAG: aie.wire(%[[S0]] : East32, %[[S1]] : West32)
// WIRE-DAG: aie.wire(%[[S0]] : North32, %[[MS]] : South32)
// WIRE-DAG: aie.wire(%[[MS]] : North32, %[[CS]] : South32)
// WIRE-DAG: aie.wire(%[[SHIM0]] : Controller32, %[[S0]] : Controller32)
// WIRE-DAG: aie.wire(%[[SHIM1]] : Control32, %[[S1]] : Control32)
// WIRE-DAG: aie.wire(%[[CORE]] : Control32, %[[CS]] : Control32)
// FLOW-LABEL: aie.device(npu3)
// FLOW-DAG: %[[SHIM0:.*]] = aie.tile(0, 0)
// FLOW-DAG: %[[SHIM1:.*]] = aie.tile(1, 0)
// FLOW-DAG: %[[CORE:.*]] = aie.tile(0, 2)
// FLOW-DAG: aie.flow(%[[CORE]], Control32 : 0, %[[SHIM0]], Controller32 : 0)
// FLOW-DAG: aie.flow(%[[SHIM1]], Control32 : 1, %[[SHIM0]], Controller32 : 1)
// FLOW-NOT: aie.flow
module {
  aie.device(npu3) {
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.switchbox(%shim0) {
      aie.connect<North32 : 0, Controller32 : 0>
      aie.connect<East32 : 0, Controller32 : 1>
    }
    aie.switchbox(%shim1) {
      aie.connect<Control32 : 1, West32 : 0>
    }
    aie.switchbox(%mem) {
      aie.connect<North32 : 0, South32 : 0>
    }
    aie.switchbox(%core) {
      aie.connect<Control32 : 0, South32 : 0>
    }
  }
}

// -----

// The control DMA engine has its own endpoint bundle. Its northbound control
// link lands on a SouthControl input and terminates at the memtile controller.
// WIRE-LABEL: aie.device(npu3_1col)
// WIRE-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
// WIRE-DAG: %[[MEM:.*]] = aie.tile(0, 1)
// WIRE: %[[SS:.*]] = aie.switchbox(%[[SHIM]])
// WIRE: %[[MS:.*]] = aie.switchbox(%[[MEM]])
// WIRE-DAG: aie.wire(%[[SHIM]] : DMAControl, %[[SS]] : DMAControl)
// WIRE-DAG: aie.wire(%[[SS]] : NorthControl, %[[MS]] : SouthControl)
// WIRE-DAG: aie.wire(%[[MEM]] : TileControl, %[[MS]] : TileControl)
// FLOW-LABEL: aie.device(npu3_1col)
// FLOW-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
// FLOW-DAG: %[[MEM:.*]] = aie.tile(0, 1)
// FLOW: aie.flow(%[[SHIM]], DMAControl : 0, %[[MEM]], TileControl : 0)
// FLOW-NOT: aie.flow
module {
  aie.device(npu3_1col) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    aie.switchbox(%shim) {
      aie.connect<DMAControl : 0, NorthControl : 0>
    }
    aie.switchbox(%mem) {
      aie.connect<SouthControl : 0, TileControl : 0>
    }
  }
}

// -----

// Recover a packet route across the narrow shim row, preserving its source,
// destination and ID. Existing fan-in tests cover sharing legality separately.
// WIRE-LABEL: aie.device(npu3)
// WIRE-DAG: %[[SHIM0:.*]] = aie.tile(0, 0)
// WIRE-DAG: %[[SHIM1:.*]] = aie.tile(1, 0)
// WIRE-DAG: %[[S0:.*]] = aie.switchbox(%[[SHIM0]])
// WIRE-DAG: %[[S1:.*]] = aie.switchbox(%[[SHIM1]])
// WIRE-DAG: aie.wire(%[[S0]] : East32, %[[S1]] : West32)
// WIRE-DAG: aie.wire(%[[SHIM0]] : Controller32, %[[S0]] : Controller32)
// WIRE-DAG: aie.wire(%[[SHIM1]] : Control32, %[[S1]] : Control32)
// FLOW-LABEL: aie.device(npu3)
// FLOW-DAG: %[[SHIM0:.*]] = aie.tile(0, 0)
// FLOW-DAG: %[[SHIM1:.*]] = aie.tile(1, 0)
// FLOW: aie.packet_flow(7) {
// FLOW-NEXT: aie.packet_source<%[[SHIM1]], Control32 : 0>
// FLOW-NEXT: aie.packet_dest<%[[SHIM0]], Controller32 : 0>
// FLOW-NEXT: }
// FLOW-NOT: aie.packet_flow
module {
  aie.device(npu3) {
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    aie.packet_flow(7) {
      aie.packet_source<%shim1, Control32 : 0>
      aie.packet_dest<%shim0, Controller32 : 0>
    }
  }
}

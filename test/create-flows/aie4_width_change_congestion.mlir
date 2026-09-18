// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows %s | FileCheck %s

// An additive width penalty alone made these legal designs exhaust the 1000
// congestion iterations. Historical overuse must scale with the penalty so a
// narrow fallback remains reachable even when wide paths exist but are full.

// Two circuits competing for the one wide exit: one must take a bridge loop.
// CHECK-LABEL: aie.device(npu3_1col)
// CHECK-DAG: aie.connect<North : {{[0-9]+}}, DMA : 2>
// CHECK-DAG: aie.connect<DMA : 0, North : 0>
// CHECK-DAG: aie.connect<Switch32 : [[BR:[0-9]+]], DMA : 3>
// CHECK-DAG: aie.connect<Switch512 : [[DOWN:[0-9]+]], Switch512 : [[BR]]>
// CHECK-DAG: aie.connect<DMA : 1, Switch32 : [[DOWN]]>

module {
  aie.device(npu3_1col) {
    %mem = aie.tile(0, 1)
    aie.switchbox(%mem) {
      aie.connect<DMA : 2, North : 1>
      aie.connect<DMA : 3, North : 2>
      aie.connect<DMA : 5, North : 3>
      aie.connect<DMA : 9, North : 4>
      aie.connect<DMA : 6, North : 5>
      aie.connect<DMA : 7, North : 6>
      aie.connect<DMA : 8, North : 7>
      aie.connect<South : 0, South : 0>
      aie.connect<South : 1, South : 1>
      aie.connect<South : 2, South : 2>
      aie.connect<South : 3, South : 3>
    }
    aie.flow(%mem, DMA : 0, %mem, DMA : 2)
    aie.flow(%mem, DMA : 1, %mem, DMA : 3)
  }
}

// -----

// Distinct sources with the same packet ID cannot merge then diverge.
// CHECK-LABEL: aie.device(npu3_1col)
// CHECK-DAG: aie.masterset(DMA : 2,
// CHECK-DAG: aie.masterset(DMA : 3,
// CHECK-DAG: aie.masterset(Switch32 :
// CHECK-DAG: aie.masterset(Switch512 :

module {
  aie.device(npu3_1col) {
    %mem = aie.tile(0, 1)
    aie.switchbox(%mem) {
      aie.connect<DMA : 2, North : 1>
      aie.connect<DMA : 3, North : 2>
      aie.connect<DMA : 5, North : 3>
      aie.connect<DMA : 9, North : 4>
      aie.connect<DMA : 6, North : 5>
      aie.connect<DMA : 7, North : 6>
      aie.connect<DMA : 8, North : 7>
      aie.connect<South : 0, South : 0>
      aie.connect<South : 1, South : 1>
      aie.connect<South : 2, South : 2>
      aie.connect<South : 3, South : 3>
    }
    aie.packet_flow(1) {
      aie.packet_source<%mem, DMA : 0>
      aie.packet_dest<%mem, DMA : 2>
    }
    aie.packet_flow(1) {
      aie.packet_source<%mem, DMA : 1>
      aie.packet_dest<%mem, DMA : 3>
    }
  }
}

// -----

// Distinct-ID packet flows must also converge under the strong preference.
// CHECK-LABEL: aie.device(npu3_1col)
// CHECK-DAG: aie.masterset(DMA : 2,
// CHECK-DAG: aie.masterset(DMA : 3,
// CHECK-DAG: aie.masterset(Switch32 :
// CHECK-DAG: aie.masterset(Switch512 :

module {
  aie.device(npu3_1col) {
    %mem = aie.tile(0, 1)
    aie.switchbox(%mem) {
      aie.connect<DMA : 2, North : 1>
      aie.connect<DMA : 3, North : 2>
      aie.connect<DMA : 5, North : 3>
      aie.connect<DMA : 9, North : 4>
      aie.connect<DMA : 6, North : 5>
      aie.connect<DMA : 7, North : 6>
      aie.connect<DMA : 8, North : 7>
      aie.connect<South : 0, South : 0>
      aie.connect<South : 1, South : 1>
      aie.connect<South : 2, South : 2>
      aie.connect<South : 3, South : 3>
    }
    aie.packet_flow(1) {
      aie.packet_source<%mem, DMA : 0>
      aie.packet_dest<%mem, DMA : 2>
    }
    aie.packet_flow(2) {
      aie.packet_source<%mem, DMA : 1>
      aie.packet_dest<%mem, DMA : 3>
    }
  }
}

// -----

// Packet and circuit traffic compete for the same limited wide capacity.
// CHECK-LABEL: aie.device(npu3_1col)
// CHECK-DAG: aie.connect<{{(North|Switch32)}} : {{[0-9]+}}, DMA : 2>
// CHECK-DAG: aie.masterset(DMA : 3,

module {
  aie.device(npu3_1col) {
    %mem = aie.tile(0, 1)
    aie.switchbox(%mem) {
      aie.connect<DMA : 2, North : 1>
      aie.connect<DMA : 3, North : 2>
      aie.connect<DMA : 5, North : 3>
      aie.connect<DMA : 9, North : 4>
      aie.connect<DMA : 6, North : 5>
      aie.connect<DMA : 7, North : 6>
      aie.connect<DMA : 8, North : 7>
      aie.connect<South : 0, South : 0>
      aie.connect<South : 1, South : 1>
      aie.connect<South : 2, South : 2>
      aie.connect<South : 3, South : 3>
    }
    aie.flow(%mem, DMA : 0, %mem, DMA : 2)
    aie.packet_flow(2) {
      aie.packet_source<%mem, DMA : 1>
      aie.packet_dest<%mem, DMA : 3>
    }
  }
}

// -----

// With two wide exits, negotiation must find two wide routes, not fall back.
// CHECK-LABEL: aie.device(npu3_1col)
// CHECK: aie.switchbox(%mem_tile_0_1) {
// CHECK-NOT: Switch32
// CHECK-NOT: Switch512
// CHECK: aie.connect<North : {{[0-9]+}}, DMA : 2>
// CHECK-NOT: Switch32
// CHECK-NOT: Switch512
// CHECK: aie.connect<North : {{[0-9]+}}, DMA : 3>
// CHECK-NOT: Switch32
// CHECK-NOT: Switch512
// CHECK: aie.switchbox(%tile_0_2) {

module {
  aie.device(npu3_1col) {
    %mem = aie.tile(0, 1)
    aie.switchbox(%mem) {
      aie.connect<DMA : 3, North : 2>
      aie.connect<DMA : 5, North : 3>
      aie.connect<DMA : 9, North : 4>
      aie.connect<DMA : 6, North : 5>
      aie.connect<DMA : 7, North : 6>
      aie.connect<DMA : 8, North : 7>
      aie.connect<South : 0, South : 0>
      aie.connect<South : 1, South : 1>
      aie.connect<South : 2, South : 2>
      aie.connect<South : 3, South : 3>
    }
    aie.flow(%mem, DMA : 0, %mem, DMA : 2)
    aie.flow(%mem, DMA : 1, %mem, DMA : 3)
  }
}

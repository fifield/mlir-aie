//===- npu3_shim_dma.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s --check-prefix=FIND

// An AIE4 shim DMA connects to the shim switchbox directly, with no shim_mux.

// CHECK-LABEL: aie.device(npu3)
// CHECK-NOT:     aie.shim_mux
// CHECK:         %[[SHIM:.*]] = aie.tile(0, 0)
// CHECK:         %[[SB00:.*]] = aie.switchbox(%[[SHIM]]) {
// CHECK-NEXT:      aie.connect<DMA : 0, North : [[UP:[0-9]+]]>
// CHECK-NEXT:      aie.connect<North : [[DOWN:[0-9]+]], DMA : 0>
// CHECK-NEXT:    }
// CHECK:         aie.switchbox(%{{.*}}) {
// CHECK-NEXT:      aie.connect<South : [[UP]], North : [[UP1:[0-9]+]]>
// CHECK-NEXT:      aie.connect<North : [[DOWN1:[0-9]+]], South : [[DOWN]]>
// CHECK-NEXT:    }
// CHECK:         aie.switchbox(%{{.*}}) {
// CHECK-NEXT:      aie.connect<South : [[UP1]], DMA : 0>
// CHECK-NEXT:      aie.connect<DMA : 0, South : [[DOWN1]]>
// CHECK-NEXT:    }
// CHECK-NOT:     aie.shim_mux
// CHECK:         aie.wire(%[[SHIM]] : DMA, %[[SB00]] : DMA)
// CHECK-NOT:     aie.shim_mux

// FIND-LABEL: aie.device(npu3)
// FIND-DAG:     aie.flow(%[[SHIM:.*]], DMA : 0, %[[CORE:.*]], DMA : 0)
// FIND-DAG:     aie.flow(%[[CORE]], DMA : 0, %[[SHIM]], DMA : 0)

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.flow(%shim, DMA : 0, %core, DMA : 0)
    aie.flow(%core, DMA : 0, %shim, DMA : 0)
  }
}

// -----

// Shim MM2S 3 has its own switchbox port. aie-find-flows does not recover it
// yet (it enumerates sending ports by S2MM count).

// CHECK-LABEL: aie.device(npu3)
// CHECK-NOT:     aie.shim_mux
// CHECK:         aie.switchbox(%{{.*}}) {
// CHECK-NEXT:      aie.connect<DMA : 3, North : {{[0-9]+}}>
// CHECK-NEXT:      aie.connect<North : {{[0-9]+}}, DMA : 1>
// CHECK-NEXT:    }
// CHECK-NOT:     aie.shim_mux

// FIND-LABEL: aie.device(npu3)
// FIND:         aie.flow(%{{.*}}, DMA : 0, %{{.*}}, DMA : 1)

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.flow(%shim, DMA : 3, %core, DMA : 1)
    aie.flow(%core, DMA : 0, %shim, DMA : 1)
  }
}

// -----

// CHECK-LABEL: aie.device(npu3)
// CHECK-NOT:     aie.shim_mux
// CHECK:         aie.switchbox(%{{.*}}) {
// CHECK-DAG:       %[[A0:.*]] = aie.amsel<{{[0-9]+}}> (0)
// CHECK-DAG:       %[[A1:.*]] = aie.amsel<{{[0-9]+}}> (0)
// CHECK-DAG:       aie.masterset(DMA : 1, %[[A1]])
// CHECK-DAG:       aie.masterset(North : {{[0-9]+}}, %[[A0]])
// CHECK-DAG:       aie.packet_rules(DMA : 1) {
// CHECK-DAG:         aie.rule(28, 0, %[[A0]])
// CHECK:         }
// CHECK-NOT:     aie.shim_mux

// FIND-LABEL: aie.device(npu3)
// FIND:         aie.packet_flow(1) {
// FIND-NEXT:      aie.packet_source<%[[SHIM:.*]], DMA : 1>
// FIND-NEXT:      aie.packet_dest<%{{.*}}, DMA : 1>
// FIND:         aie.packet_flow(2) {
// FIND-NEXT:      aie.packet_source<%[[SHIM]], DMA : 1>
// FIND-NEXT:      aie.packet_dest<%{{.*}}, DMA : 1>
// FIND:         aie.packet_flow(3) {
// FIND-NEXT:      aie.packet_source<%{{.*}}, DMA : 0>
// FIND-NEXT:      aie.packet_dest<%[[SHIM]], DMA : 1>

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    %core2 = aie.tile(0, 3)
    aie.packet_flow(1) {
      aie.packet_source<%shim, DMA : 1>
      aie.packet_dest<%core, DMA : 1>
    }
    aie.packet_flow(2) {
      aie.packet_source<%shim, DMA : 1>
      aie.packet_dest<%core2, DMA : 1>
    }
    aie.packet_flow(3) {
      aie.packet_source<%core, DMA : 0>
      aie.packet_dest<%shim, DMA : 1>
    }
  }
}

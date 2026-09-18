//===- aie4_width_change_penalty.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows %s | FileCheck %s

// A wide detour is available for this otherwise restricted DMA loopback.
// Do not downsize it merely to save hops or win a tie.
// CHECK-LABEL: aie.device(npu3_1col)
// CHECK: aie.switchbox(%mem_tile_0_1) {
// CHECK-NOT: Switch32
// CHECK-NOT: Switch512
// CHECK: aie.connect<North : [[UP:[0-9]+]], DMA : 2>
// CHECK: aie.connect<DMA : 0, North : [[OUT:[0-9]+]]>
// CHECK: aie.switchbox(%tile_0_2) {
// CHECK: aie.connect<South : [[OUT]], South : [[UP]]>

module {
  aie.device(npu3_1col) {
    %mem = aie.tile(0, 1)
    aie.flow(%mem, DMA : 0, %mem, DMA : 2)
  }
}

// -----

// Intentional 32 -> 512: one crossing is required for shim trace capture.
// Its fixed penalty is common to all candidate paths.
// CHECK-LABEL: aie.device(npu3_1col)
// CHECK: aie.masterset(DMA : 1, %[[DMA:[a-zA-Z0-9_]+]])
// CHECK: aie.masterset(Switch512 : [[BR:[0-9]+]], %[[TRACE:[a-zA-Z0-9_]+]])
// CHECK: aie.packet_rules(Trace32 : 0) {
// CHECK-NEXT: aie.rule(31, 7, %[[TRACE]])
// CHECK: aie.packet_rules(Switch32 : [[BR]]) {
// CHECK-NEXT: aie.rule(31, 7, %[[DMA]])

module {
  aie.device(npu3_1col) {
    %shim = aie.tile(0, 0)
    aie.packet_flow(7) {
      aie.packet_source<%shim, Trace32 : 0>
      aie.packet_dest<%shim, DMA : 1>
    } {keep_pkt_header = true}
  }
}

// -----

// Intentional 512 -> 32 works as well.
// CHECK-LABEL: aie.device(npu3_1col)
// CHECK: aie.masterset(Switch32 : [[BR:[0-9]+]], %[[DMA:[a-zA-Z0-9_]+]])
// CHECK: aie.masterset(Controller32 : 0, %[[CTRL:[a-zA-Z0-9_]+]])
// CHECK: aie.packet_rules(DMA : 0) {
// CHECK-NEXT: aie.rule(31, 7, %[[DMA]])
// CHECK: aie.packet_rules(Switch512 : [[BR]]) {
// CHECK-NEXT: aie.rule(31, 7, %[[CTRL]])

module {
  aie.device(npu3_1col) {
    %shim = aie.tile(0, 0)
    aie.packet_flow(7) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%shim, Controller32 : 0>
    }
  }
}

// -----

// All wide exits are reserved, so a narrow fallback must remain possible.
// CHECK-LABEL: aie.device(npu3_1col)
// CHECK: aie.connect<Switch32 : [[BR:[0-9]+]], DMA : 2>
// CHECK: aie.connect<Switch512 : [[DOWN:[0-9]+]], Switch512 : [[BR]]>
// CHECK: aie.connect<DMA : 0, Switch32 : [[DOWN]]>

module {
  aie.device(npu3_1col) {
    %mem = aie.tile(0, 1)
    aie.switchbox(%mem) {
      aie.connect<DMA : 4, North : 0>
      aie.connect<DMA : 1, North : 1>
      aie.connect<DMA : 2, North : 2>
      aie.connect<DMA : 3, North : 3>
      aie.connect<DMA : 9, North : 4>
      aie.connect<DMA : 5, North : 5>
      aie.connect<DMA : 6, North : 6>
      aie.connect<DMA : 7, North : 7>
      aie.connect<South : 0, South : 0>
      aie.connect<South : 1, South : 1>
      aie.connect<South : 2, South : 2>
      aie.connect<South : 3, South : 3>
    }
    aie.flow(%mem, DMA : 0, %mem, DMA : 2)
  }
}

// -----

// With the fixed penalty, IDs 1/2 stay wide and ID 3 crosses only once,
// at the shim. Sharing a source with an intentional crossing must not make
// the data destinations pay for an unnecessary narrow detour.
// CHECK-LABEL: aie.device(npu3_1col)
// CHECK: aie.switchbox(%mem_tile_0_1) {
// CHECK: aie.masterset(DMA : 2, %[[DMA_AMSEL:[a-zA-Z0-9_]+]])
// CHECK: aie.masterset(South : [[CTRL:[0-9]+]], %[[CTRL_AMSEL:[a-zA-Z0-9_]+]])
// CHECK: aie.masterset(North : [[DATA:[0-9]+]], %[[DATA_AMSEL:[a-zA-Z0-9_]+]])
// CHECK-NOT: Switch32
// CHECK-NOT: Switch512
// CHECK: aie.packet_rules(DMA : 0) {
// CHECK-NEXT: aie.rule(31, 3, %[[CTRL_AMSEL]])
// CHECK-NEXT: aie.rule(31, 1, %[[DATA_AMSEL]])
// CHECK-NEXT: aie.rule(31, 2, %[[DATA_AMSEL]])
// CHECK-NEXT: }
// CHECK: aie.packet_rules(North : [[DATA]]) {
// CHECK-NEXT: aie.rule(28, 0, %[[DMA_AMSEL]])
// CHECK: aie.switchbox(%shim_noc_tile_0_0) {
// CHECK: aie.masterset(Switch32 : [[BR:[0-9]+]], %[[BR_AMSEL:[a-zA-Z0-9_]+]])
// CHECK: aie.masterset(Controller32 : 0, %[[DEST_AMSEL:[a-zA-Z0-9_]+]])
// CHECK: aie.packet_rules(North : [[CTRL]]) {
// CHECK-NEXT: aie.rule(31, 3, %[[BR_AMSEL]])
// CHECK: aie.packet_rules(Switch512 : [[BR]]) {
// CHECK-NEXT: aie.rule(31, 3, %[[DEST_AMSEL]])

module {
  aie.device(npu3_1col) {
    %mem = aie.tile(0, 1)
    %shim = aie.tile(0, 0)
    aie.packet_flow(1) {
      aie.packet_source<%mem, DMA : 0>
      aie.packet_dest<%mem, DMA : 2>
    }
    aie.packet_flow(2) {
      aie.packet_source<%mem, DMA : 0>
      aie.packet_dest<%mem, DMA : 2>
    }
    aie.packet_flow(3) {
      aie.packet_source<%mem, DMA : 0>
      aie.packet_dest<%shim, Controller32 : 0>
    }
  }
}

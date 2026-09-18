//===- aie4_bridge_packet_fanout.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// IDs 1 and 2 take a local 512 -> 32 -> 512 loopback to DMA:2. ID 3 shares
// the first bridge crossing but leaves the 32-bit switch on South32 instead.
// An output must not be treated as the same-named input during reachability:
// Switch512's output feeds Switch32's input, not Switch512's own input.
// Otherwise ID 3 is also broadcast back across the bridge, where DMA:2's
// combined rule (mask 28, value 0) accepts it and corrupts the DMA stream.
// Reserved wide exits make this detour necessary under the fixed width policy.

// CHECK-LABEL: aie.device(npu3_1col)
// CHECK: aie.switchbox(%shim_noc_tile_0_0) {
// CHECK: aie.masterset(Controller32 : 0, %[[CONTROLLER_AMSEL:[a-zA-Z0-9_]+]])
// CHECK: aie.packet_rules(North32 : [[SOUTH:[0-9]+]]) {
// CHECK-NEXT: aie.rule(31, 3, %[[CONTROLLER_AMSEL]])
// CHECK: aie.switchbox(%mem_tile_0_1) {
// CHECK: aie.masterset(DMA : 2, %[[DMA_AMSEL:[a-zA-Z0-9_]+]])
// CHECK: aie.masterset(Switch32 : [[DOWN:[0-9]+]], %[[SRC_AMSEL:[a-zA-Z0-9_]+]])
// CHECK: aie.masterset(South32 : [[SOUTH]], %[[CTRL_AMSEL:[a-zA-Z0-9_]+]])
// The bridge master must have exactly one amsel, excluding CTRL_AMSEL.
// CHECK: aie.masterset(Switch512 : [[UP:[0-9]+]], %[[LOCAL_AMSEL:[a-zA-Z0-9_]+]])
// CHECK: aie.packet_rules(DMA : 0) {
// CHECK-NEXT: aie.rule(28, 0, %[[SRC_AMSEL]])
// CHECK-NEXT: }
// CHECK: aie.packet_rules(Switch512 : [[DOWN]]) {
// CHECK-NEXT: aie.rule(31, 3, %[[CTRL_AMSEL]])
// CHECK-NEXT: aie.rule(31, 1, %[[LOCAL_AMSEL]])
// CHECK-NEXT: aie.rule(31, 2, %[[LOCAL_AMSEL]])
// CHECK-NEXT: }
// CHECK: aie.packet_rules(Switch32 : [[UP]]) {
// CHECK-NEXT: aie.rule(28, 0, %[[DMA_AMSEL]])
// CHECK-NEXT: }

module {
  aie.device(npu3_1col) {
    %mem = aie.tile(0, 1)
    %shim = aie.tile(0, 0)
    // Reserve every wide exit reachable by DMA:0. The source must cross the
    // bridge, so the return leg and the Controller32 branch share a narrow
    // input without relying on a cost override.
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

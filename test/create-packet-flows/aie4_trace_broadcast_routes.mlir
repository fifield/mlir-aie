// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-create-pathfinder-flows %s > %t
// RUN: FileCheck %s < %t
// RUN: FileCheck %s --check-prefix=WIDTH < %t
// RUN: aie-opt --aie-find-flows %t | FileCheck %s --check-prefix=FLOW
// RUN: aie-opt --aie-create-pathfinder-flows --convert-aie-to-transaction %s -o /dev/null

// Trace ID 3 crosses to the wide switch at the core, then branches at memory.
// One leg reaches DMA6 directly; the other takes a wide shim loop to DMA3.
// Both destinations receive the trace with one width crossing on each path.
// CHECK: aie.switchbox(%mem_tile_0_1)
// CHECK: aie.masterset(DMA : 3, %[[DMA3:[a-zA-Z0-9_]+]])
// CHECK: aie.masterset(DMA : 6, %{{[a-zA-Z0-9_]+}}, %[[FANOUT:[a-zA-Z0-9_]+]])
// CHECK: aie.masterset(South : [[LOOP:[0-9]+]], %[[FANOUT]])
// CHECK: aie.packet_rules(North : [[WIDE:[0-9]+]]) {
// CHECK-NEXT: aie.rule(31, 3, %[[FANOUT]])
// CHECK: aie.packet_rules(South : [[LOOP]]) {
// CHECK-NEXT: aie.rule(31, 3, %[[DMA3]])
// CHECK: aie.switchbox(%tile_0_2)
// CHECK: aie.masterset(South : [[WIDE]], %[[CORE_WIDE:[a-zA-Z0-9_]+]])
// CHECK: aie.masterset(Switch512 : [[BR:[0-9]+]], %[[BRIDGE:[a-zA-Z0-9_]+]])
// CHECK: aie.packet_rules(Trace32 : 0) {
// CHECK-NEXT: aie.rule(31, 3, %[[BRIDGE]])
// CHECK: aie.packet_rules(Switch32 : [[BR]]) {
// CHECK-NEXT: aie.rule(31, 3, %[[CORE_WIDE]])

// Once ID 3 crosses at its source, memory must not add a narrow detour.
// WIDTH-LABEL: aie.switchbox(%mem_tile_0_1)
// WIDTH-NOT: aie.masterset(Switch
// WIDTH-NOT: aie.masterset(North32
// WIDTH-NOT: aie.masterset(South32
// WIDTH: aie.switchbox(%tile_0_2)

// Recover every original relation, including both trace broadcast legs.
// FLOW: aie.packet_flow(10) {
// FLOW-NEXT: aie.packet_source<%tile_0_2, DMA : 0>
// FLOW-NEXT: aie.packet_dest<%shim_noc_tile_1_0, Controller32 : 0>
// FLOW-NEXT: }
// FLOW-NEXT: aie.packet_flow(11) {
// FLOW-NEXT: aie.packet_source<%tile_0_3, DMA : 0>
// FLOW-NEXT: aie.packet_dest<%mem_tile_0_1, DMA : 3>
// FLOW-NEXT: }
// FLOW-NEXT: aie.packet_flow(12) {
// FLOW-NEXT: aie.packet_source<%shim_noc_tile_0_0, Trace32 : 0>
// FLOW-NEXT: aie.packet_dest<%mem_tile_0_1, DMA : 6>
// FLOW-NEXT: }
// FLOW-NEXT: aie.packet_flow(3) {
// FLOW-NEXT: aie.packet_source<%tile_0_2, Trace32 : 0>
// FLOW-NEXT: aie.packet_dest<%mem_tile_0_1, DMA : 3>
// FLOW-NEXT: }
// FLOW-NEXT: aie.packet_flow(3) {
// FLOW-NEXT: aie.packet_source<%tile_0_2, Trace32 : 0>
// FLOW-NEXT: aie.packet_dest<%mem_tile_0_1, DMA : 6>
// FLOW-NEXT: }
// FLOW-NEXT: }

module {
  aie.device(npu3_2col) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    %upper = aie.tile(0, 3)
    %shim1 = aie.tile(1, 0)
    aie.packet_flow(11) {
      aie.packet_source<%upper, DMA : 0>
      aie.packet_dest<%mem, DMA : 3>
    }
    aie.packet_flow(12) {
      aie.packet_source<%shim, Trace32 : 0>
      aie.packet_dest<%mem, DMA : 6>
    }
    aie.packet_flow(3) {
      aie.packet_source<%core, Trace32 : 0>
      aie.packet_dest<%mem, DMA : 3>
      aie.packet_dest<%mem, DMA : 6>
    }
    aie.packet_flow(10) {
      aie.packet_source<%core, DMA : 0>
      aie.packet_dest<%shim1, Controller32 : 0>
    }
  }
}

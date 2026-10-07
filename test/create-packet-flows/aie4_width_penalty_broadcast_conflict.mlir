// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-create-pathfinder-flows %s > %t
// RUN: FileCheck %s --implicit-check-not='aie.masterset(Switch' < %t
// RUN: aie-opt --aie-find-flows %t | FileCheck %s --check-prefix=FLOW

// The old greedy allocator pinned DMA3 and DMA6 to incompatible arbiters and
// had to reject this multicast to avoid silently dropping a leg. The planner
// now finds legal wide detours: ID 3 fans out to North/South on one amsel,
// then reaches both destinations. No additional width crossing is necessary.
// CHECK: aie.switchbox(%mem_tile_0_1)
// CHECK: aie.masterset(DMA : 3,
// CHECK: aie.masterset(DMA : 6,
// CHECK: aie.masterset(South : [[DOWN:[0-9]+]], %[[FANOUT:[a-zA-Z0-9_]+]])
// CHECK: aie.masterset(North : [[UP:[0-9]+]], %[[FANOUT]])
// CHECK: aie.packet_rules(Switch32 : 0) {
// CHECK-NEXT: aie.rule(31, 3, %[[FANOUT]])

// Flow lifting must recover both multicast legs and both original unicasts.
// FLOW: aie.packet_flow(11) {
// FLOW-NEXT: aie.packet_source<%mem_tile_0_1, North : 1>
// FLOW-NEXT: aie.packet_dest<%mem_tile_0_1, DMA : 3>
// FLOW-NEXT: }
// FLOW-NEXT: aie.packet_flow(3) {
// FLOW-NEXT: aie.packet_source<%mem_tile_0_1, Switch32 : 0>
// FLOW-NEXT: aie.packet_dest<%mem_tile_0_1, DMA : 6>
// FLOW-NEXT: }
// FLOW-NEXT: aie.packet_flow(3) {
// FLOW-NEXT: aie.packet_source<%mem_tile_0_1, Switch32 : 0>
// FLOW-NEXT: aie.packet_dest<%mem_tile_0_1, DMA : 3>
// FLOW-NEXT: }
// FLOW-NEXT: aie.packet_flow(12) {
// FLOW-NEXT: aie.packet_source<%mem_tile_0_1, North : 2>
// FLOW-NEXT: aie.packet_dest<%mem_tile_0_1, DMA : 6>
// FLOW-NEXT: }
// FLOW-NEXT: }

module {
  aie.device(npu3_1col) {
    %mem = aie.tile(0, 1)
    aie.packet_flow(11) {
      aie.packet_source<%mem, North : 1>
      aie.packet_dest<%mem, DMA : 3>
    }
    aie.packet_flow(12) {
      aie.packet_source<%mem, North : 2>
      aie.packet_dest<%mem, DMA : 6>
    }
    aie.packet_flow(3) {
      aie.packet_source<%mem, Switch32 : 0>
      aie.packet_dest<%mem, DMA : 3>
      aie.packet_dest<%mem, DMA : 6>
    }
  }
}

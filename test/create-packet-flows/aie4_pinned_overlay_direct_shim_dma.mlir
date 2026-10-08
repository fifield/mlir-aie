// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows %s | FileCheck %s --implicit-check-not=aie.shim_mux
// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows="circuit-switch-hops=false" %s | FileCheck %s --implicit-check-not=aie.shim_mux

// A reload keeps the overlay's route and amsel. On AIE4 its shim rule lives
// directly on DMA:0, not the legacy South:3 mux input. Compare the overlay
// routed alone with the same prioritized control packet plus an unrelated
// application packet. The application must leave its pinned amsel and rule
// intact. Both tile controllers are reached so a reload can configure them.

// CHECK-LABEL: module @alone
// CHECK: aie.switchbox(%shim_noc_tile_0_0)
// CHECK: %[[ALONE:.*]] = aie.amsel<[[ARBITER:[0-9]+]]> ([[MSEL:[0-9]+]])
// CHECK: aie.masterset(NorthControl : [[CHANNEL:[0-9]+]], %[[ALONE]]) {aie.is_ctrl_pkt_overlay}
// CHECK: aie.packet_rules(DMA : 0) {
// CHECK-NEXT: aie.rule(31, 7, %[[ALONE]])

module @alone {
  aie.device(npu3_1col) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    aie.packet_flow(7) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%shim, TileControl : 0>
      aie.packet_dest<%mem, TileControl : 0>
    } {priority_route = true}
  }
}

// -----

// CHECK-LABEL: module @reload
// CHECK: aie.switchbox(%shim_noc_tile_0_0)
// CHECK: %[[RELOAD:.*]] = aie.amsel<[[ARBITER]]> ([[MSEL]])
// CHECK: aie.masterset(NorthControl : [[CHANNEL]], %[[RELOAD]]) {aie.is_ctrl_pkt_overlay}
// CHECK: aie.packet_rules(DMA : 1) {
// CHECK-NEXT: aie.rule(31, 8,
// CHECK: aie.packet_rules(DMA : 0) {
// CHECK-NEXT: aie.rule(31, 7, %[[RELOAD]])
// CHECK-NEXT: } {aie.is_ctrl_pkt_overlay}

module @reload {
  aie.device(npu3_1col) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    aie.packet_flow(7) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%shim, TileControl : 0>
      aie.packet_dest<%mem, TileControl : 0>
    } {priority_route = true}
    aie.packet_flow(8) {
      aie.packet_source<%shim, DMA : 1>
      aie.packet_dest<%mem, DMA : 0>
    }
  } {has_ctrl_pkt_overlay = true}
}

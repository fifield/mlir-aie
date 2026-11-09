// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -aie-assign-tile-controller-ids -aie-generate-column-control-overlay="configuration=true adaptive=false" | FileCheck %s --implicit-check-not=aie.packet_flow --implicit-check-not="DMAControl : 1"
// RUN: aie-opt %s -aie-assign-tile-controller-ids -aie-generate-column-control-overlay="configuration=true" | FileCheck %s --implicit-check-not=aie.packet_flow --implicit-check-not="DMAControl : 1"

// CHECK-LABEL: aie.device(npu3) {
// CHECK: %[[SHIM:.*]] = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_id = 15>}
// CHECK: %[[MEM:.*]] = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_id = 26>}
// CHECK: %[[CORE2:.*]] = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_id = 27>}
// CHECK: %[[CORE3:.*]] = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_id = 29>}
// CHECK: %[[CORE4:.*]] = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_id = 30>}
// CHECK: %[[CORE5:.*]] = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_id = 31>}
// CHECK: overlay = @ctrl_pkt_overlay
// CHECK-LABEL: aie.device(npu3) @ctrl_pkt_overlay
// CHECK: %[[OV5:.*]] = aie.tile(0, 5)
// CHECK: aie.switchbox(%[[OV5]])
// CHECK: aie.masterset(TileControl : 0, {{.*}}) {aie.is_ctrl_pkt_overlay, keep_pkt_header = false}
// CHECK: aie.rule(31, 31, {{.*}}) {aie.is_ctrl_pkt_overlay}
// CHECK: %[[OV4:.*]] = aie.tile(0, 4)
// CHECK: aie.switchbox(%[[OV4]])
// CHECK: aie.masterset(TileControl : 0, {{.*}}) {aie.is_ctrl_pkt_overlay, keep_pkt_header = false}
// CHECK: aie.rule(31, 30, {{.*}}) {aie.is_ctrl_pkt_overlay}
// CHECK: %[[OV3:.*]] = aie.tile(0, 3)
// CHECK: aie.switchbox(%[[OV3]])
// CHECK: aie.masterset(TileControl : 0, {{.*}}) {aie.is_ctrl_pkt_overlay, keep_pkt_header = false}
// CHECK: aie.rule(31, 29, {{.*}}) {aie.is_ctrl_pkt_overlay}
// CHECK: %[[OV2:.*]] = aie.tile(0, 2)
// CHECK: aie.switchbox(%[[OV2]])
// CHECK: aie.masterset(TileControl : 0, {{.*}}) {aie.is_ctrl_pkt_overlay, keep_pkt_header = false}
// CHECK: aie.rule(31, 27, {{.*}}) {aie.is_ctrl_pkt_overlay}
// CHECK: %[[OV1:.*]] = aie.tile(0, 1)
// CHECK: aie.switchbox(%[[OV1]])
// CHECK: aie.masterset(TileControl : 0, {{.*}}) {aie.is_ctrl_pkt_overlay, keep_pkt_header = false}
// CHECK: aie.rule(31, 26, {{.*}}) {aie.is_ctrl_pkt_overlay}
// CHECK: %[[OVSHIM:.*]] = aie.tile(0, 0)
// CHECK: aie.switchbox(%[[OVSHIM]])
// CHECK: aie.connect<DMAControl : 0, {{.*}}> {aie.is_ctrl_pkt_overlay}
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%[[OVSHIM]], ControlMM2S, 0)

aie.device(npu3) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  %tile_0_2 = aie.tile(0, 2)
  %tile_0_3 = aie.tile(0, 3)
  %tile_0_4 = aie.tile(0, 4)
  %tile_0_5 = aie.tile(0, 5)
}

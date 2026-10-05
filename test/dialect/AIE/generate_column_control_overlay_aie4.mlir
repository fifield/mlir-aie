// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s -aie-assign-tile-controller-ids -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" | FileCheck %s

// CHECK-LABEL: aie.device(npu3) {
// CHECK: %[[SHIM:.*]] = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
// CHECK: %[[MEM:.*]] = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
// CHECK: %[[CORE2:.*]] = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
// CHECK: %[[CORE3:.*]] = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
// CHECK: %[[CORE4:.*]] = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
// CHECK: %[[CORE5:.*]] = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
// CHECK: aie.packet_flow(26) {
// CHECK:   aie.packet_source<%[[SHIM]], DMAControl : 0>
// CHECK:   aie.packet_dest<%[[MEM]], TileControl : 0>
// CHECK: } {keep_pkt_header = false, priority_route = true}
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_ctrl_mm2s_chan0(%[[SHIM]], ControlMM2S, 0)
// CHECK: aie.packet_flow(27) {
// CHECK:   aie.packet_source<%[[SHIM]], DMAControl : 0>
// CHECK:   aie.packet_dest<%[[CORE2]], TileControl : 0>
// CHECK: } {keep_pkt_header = false, priority_route = true}
// CHECK: aie.packet_flow(29) {
// CHECK:   aie.packet_source<%[[SHIM]], DMAControl : 1>
// CHECK:   aie.packet_dest<%[[CORE3]], TileControl : 0>
// CHECK: } {keep_pkt_header = false, priority_route = true}
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_ctrl_mm2s_chan1(%[[SHIM]], ControlMM2S, 1)
// CHECK: aie.packet_flow(30) {
// CHECK:   aie.packet_source<%[[SHIM]], DMAControl : 1>
// CHECK:   aie.packet_dest<%[[CORE4]], TileControl : 0>
// CHECK: } {keep_pkt_header = false, priority_route = true}
// CHECK: aie.packet_flow(31) {
// CHECK:   aie.packet_source<%[[SHIM]], DMAControl : 1>
// CHECK:   aie.packet_dest<%[[CORE5]], TileControl : 0>
// CHECK: } {keep_pkt_header = false, priority_route = true}

aie.device(npu3) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  %tile_0_2 = aie.tile(0, 2)
  %tile_0_3 = aie.tile(0, 3)
  %tile_0_4 = aie.tile(0, 4)
  %tile_0_5 = aie.tile(0, 5)
}

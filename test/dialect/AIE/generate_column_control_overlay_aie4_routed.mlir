// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --split-input-file --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" --aie-create-pathfinder-flows="circuit-switch-hops=false" > %t.once
// RUN: aie-opt %t.once --split-input-file --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" > %t.twice
// RUN: diff %t.once %t.twice
// RUN: FileCheck %s --implicit-check-not=aie.packet_flow < %t.twice
// Earlier Medusa output used an unqualified Boolean overlay marker.
// RUN: sed 's/{aie.is_ctrl_pkt_overlay}/{is_ctrl_pkt_overlay = true}/g' %s > %t.legacy.mlir
// RUN: aie-opt %t.legacy.mlir --split-input-file --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" --aie-create-pathfinder-flows="circuit-switch-hops=false" > %t.legacy.once
// RUN: aie-opt %t.legacy.once --split-input-file --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" > %t.legacy.twice
// RUN: diff %t.legacy.once %t.legacy.twice
// RUN: FileCheck %s --implicit-check-not=aie.packet_flow < %t.legacy.twice

// Both control engines must be recognized after routing. The row mapping
// uses two ControlMM2S channels, independently of the four ordinary MM2S
// channels. AIE4's shim self-configuration and placement-owned TCT returns
// must not cause this pass to regenerate already-routed request flows.
// CHECK-LABEL: module @generated {
// CHECK-DAG: aie.connect<DMAControl : 0,
// CHECK-DAG: aie.connect<DMAControl : 1,
// CHECK-DAG: aie.shim_dma_allocation @ctrlpkt_col0_ctrl_mm2s_chan0({{.*}}, ControlMM2S, 0)
// CHECK-DAG: aie.shim_dma_allocation @ctrlpkt_col0_ctrl_mm2s_chan1({{.*}}, ControlMM2S, 1)
module @generated {
  aie.device(npu3_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 3)
  }
}

// -----

// This overlay request takes both same-tile width bridges and a narrow
// vertical wire. Its existing control DMA allocation and installed endpoint
// must be recognized before any additional flow is declared.
// CHECK-LABEL: module @bridged {
// CHECK: aie.connect<DMAControl : 0, Switch32 : 0>
// CHECK: aie.connect<Switch512 : 0, North32 : 0>
// CHECK: aie.connect<South32 : 0, Switch512 : 0>
// CHECK: aie.masterset(TileControl : 0,
module @bridged {
  aie.device(npu3_1col) {
    %shim = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %shim_sw = aie.switchbox(%shim) {
      aie.connect<DMAControl : 0, Switch32 : 0> {aie.is_ctrl_pkt_overlay}
      aie.connect<Switch512 : 0, North32 : 0> {aie.is_ctrl_pkt_overlay}
    }
    %mem_sw = aie.switchbox(%mem) {
      aie.connect<South32 : 0, Switch512 : 0> {aie.is_ctrl_pkt_overlay}
      %a = aie.amsel<0> (0)
      aie.masterset(TileControl : 0, %a) {aie.is_ctrl_pkt_overlay}
      aie.packet_rules(Switch32 : 0) {
        aie.rule(31, 26, %a) {aie.is_ctrl_pkt_overlay}
      }
    }
    aie.shim_dma_allocation @ctrlpkt_col0_ctrl_mm2s_chan0(%shim, ControlMM2S, 0)
  }
}

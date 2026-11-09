// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --split-input-file --aie-generate-column-control-overlay="configuration=true adaptive=false" --aie-create-pathfinder-flows="circuit-switch-hops=false" > %t.once
// RUN: aie-opt %t.once --split-input-file --aie-generate-column-control-overlay="configuration=true adaptive=false" > %t.twice
// RUN: diff %t.once %t.twice
// RUN: FileCheck %s --implicit-check-not=aie.packet_flow < %t.twice
// Generation is idempotent for an application referencing a standalone overlay.
// AIE4 uses one ControlMM2S channel for the column and placement-owned TCT returns.
// CHECK-LABEL: module @generated {
// CHECK-DAG: aie.connect<DMAControl : 0,
// CHECK-DAG: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0({{.*}}, ControlMM2S, 0)
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
  aie.device(npu3_1col) @app {
    %shim = aie.tile(0, 0)
  } {overlay = @transport}
  aie.device(npu3_1col) @transport {
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
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%shim, ControlMM2S, 0)
  } {aie.is_ctrl_pkt_overlay}
}

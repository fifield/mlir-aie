//===- aie4_control_overlay_fan_in.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Regression test for same-packet-id FAN-IN, the legal counterpart to the
// merge-then-fanout bug guarded by same_id_shared_link.mlir (#3472).
//
// This is emitAIE4ControlWiring()'s column control overlay: every shim tile
// funnels its Control32:0/1 token-return ports into the first shim's single
// Controller32:0, using that shim's controller_id as the shared packet id
// (the destination matches on the source *tile*'s id, not the channel).
//
// Do not relax this into a "does it route" check: the point is that the two
// same-id streams share one amsel that drives exactly ONE masterset port.

// CHECK-LABEL: aie.device(npu3)

// Column 1's shim: Control32:0/1 both carry id 1, merging onto one amsel
// that drives the single westbound master port -- the fan-in that must stay
// legal.
// CHECK:      %[[shim1:.*]] = aie.tile(1, 0)
// CHECK:      aie.switchbox(%[[shim1]]) {
// CHECK-NEXT:   %[[c0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   aie.masterset(West32 : 0, %[[c0]])
// CHECK-NEXT:   aie.packet_rules(Control32 : 1) {
// CHECK-NEXT:     aie.rule(31, 1, %[[c0]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(Control32 : 0) {
// CHECK-NEXT:     aie.rule(31, 1, %[[c0]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
  aie.device(npu3) {
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
    // Column 0 control return (id 0), local to the first shim.
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_0_0, Control32 : 0>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_0_0, Control32 : 1>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }
    // Column 1 control return (id 1), crossing the shim row into column 0.
    aie.packet_flow(1) {
      aie.packet_source<%shim_noc_tile_1_0, Control32 : 0>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%shim_noc_tile_1_0, Control32 : 1>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }
  }
}

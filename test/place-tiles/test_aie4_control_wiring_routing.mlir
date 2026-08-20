//===- test_aie4_control_wiring_routing.mlir -------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// End-to-end companion to test_aie4_control_wiring.mlir. That test stops at
// --aie-place-tiles and only checks emitAIE4ControlWiring()'s generated
// packet_flow ops; it never reaches the router, so it can't catch a routing
// regression. This runs the full path -- wiring then pathfinder -- and
// checks the overlay actually routes.
//
// Same invariant as aie4_control_overlay_fan_in.mlir, reached through the
// real pass instead of hand-authored IR: same-id streams merge onto ONE
// amsel driving ONE masterset port. Keep the CHECK-NEXT chains intact -- they
// pin "exactly one masterset per switchbox".

// RUN: aie-opt --aie-place-tiles --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL: @two_columns_routed

// Column 1's shim: Control32:0/1 carry controller_id 1, fan in to one
// westbound master port.
// CHECK:      %[[SHIM1:.*]] = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
// CHECK:      aie.switchbox(%[[SHIM1]]) {
// CHECK-NEXT:   %[[A1:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   aie.masterset(West32 : 0, %[[A1]])
// CHECK-NEXT:   aie.packet_rules(Control32 : 1) {
// CHECK-NEXT:     aie.rule(31, 1, %[[A1]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(Control32 : 0) {
// CHECK-NEXT:     aie.rule(31, 1, %[[A1]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

// Column 0's shim: the id-1 stream from the east joins its own two id-0
// Control32 ports, all three fan in to Controller32:0 on one amsel.
// CHECK:      %[[SHIM0:.*]] = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
// CHECK:      aie.switchbox(%[[SHIM0]]) {
// CHECK-NEXT:   %[[A0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   aie.masterset(Controller32 : 0, %[[A0]])
// CHECK-NEXT:   aie.packet_rules(East32 : 0) {
// CHECK-NEXT:     aie.rule(31, 1, %[[A0]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(Control32 : 1) {
// CHECK-NEXT:     aie.rule(31, 0, %[[A0]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(Control32 : 0) {
// CHECK-NEXT:     aie.rule(31, 0, %[[A0]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

// No hand-written controller_id, so emitAIE4ControlWiring() supplies the
// overlay -- same @two_columns device as test_aie4_control_wiring.mlir.
module @two_columns_routed {
  aie.device(npu3) {
    %s0 = aie.logical_tile<ShimNOCTile>(0, 0)
    %m0 = aie.logical_tile<MemTile>(0, 1)
    %s1 = aie.logical_tile<ShimNOCTile>(1, 0)
    %m1 = aie.logical_tile<MemTile>(1, 1)
    aie.objectfifo @of0(%s0, {%m0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of1(%s1, {%m1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}

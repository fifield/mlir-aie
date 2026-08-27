//===- test_aie4_control_wiring.mlir ---------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// On AIE4 (npu3/npu3z), --aie-place-tiles emits the control flows after tile
// placement: every shim NOC/PL tile used by the design gets a controller_id
// PacketInfoAttr and a Control32:0 / Control32:1 -> Controller32:0 packet flow
// back to the first shim's Controller32 channel 0.

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// A single shim (memcpy-style shim->mem->shim) is merged to one physical tile
// and gets exactly one controller_id and two Control32 packet flows.
// CHECK-LABEL: @single_shim
module @single_shim {
  aie.device(npu3) {
    // CHECK: %[[SHIM:.*]] = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %m = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%m, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // CHECK: aie.packet_flow(0) {
    // CHECK-NEXT: aie.packet_source<%[[SHIM]], Control32 : 0>
    // CHECK-NEXT: aie.packet_dest<%[[SHIM]], Controller32 : 0>
    // CHECK: aie.packet_flow(0) {
    // CHECK-NEXT: aie.packet_source<%[[SHIM]], Control32 : 1>
    // CHECK-NEXT: aie.packet_dest<%[[SHIM]], Controller32 : 0>
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Two shim columns each get a unique controller_id (pkt_id = column ordinal),
// and all flows are destined to the first shim's Controller32 channel 0.
// CHECK-LABEL: @two_columns
module @two_columns {
  aie.device(npu3) {
    // CHECK-DAG: %[[SHIM0:.*]] = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    // CHECK-DAG: %[[SHIM1:.*]] = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
    %s0 = aie.logical_tile<ShimNOCTile>(0, 0)
    %m0 = aie.logical_tile<MemTile>(0, 1)
    %s1 = aie.logical_tile<ShimNOCTile>(1, 0)
    %m1 = aie.logical_tile<MemTile>(1, 1)
    aie.objectfifo @of0(%s0, {%m0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of1(%s1, {%m1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // CHECK: aie.packet_flow(0) {
    // CHECK-NEXT: aie.packet_source<%[[SHIM0]], Control32 : 0>
    // CHECK-NEXT: aie.packet_dest<%[[SHIM0]], Controller32 : 0>
    // CHECK: aie.packet_flow(0) {
    // CHECK-NEXT: aie.packet_source<%[[SHIM0]], Control32 : 1>
    // CHECK-NEXT: aie.packet_dest<%[[SHIM0]], Controller32 : 0>
    // CHECK: aie.packet_flow(1) {
    // CHECK-NEXT: aie.packet_source<%[[SHIM1]], Control32 : 0>
    // CHECK-NEXT: aie.packet_dest<%[[SHIM0]], Controller32 : 0>
    // CHECK: aie.packet_flow(1) {
    // CHECK-NEXT: aie.packet_source<%[[SHIM1]], Control32 : 1>
    // CHECK-NEXT: aie.packet_dest<%[[SHIM0]], Controller32 : 0>
  }
}

// -----

// A design that is already wired (controller_id present, e.g. the hand-written
// *_placed.py examples) is left untouched: no extra controller_id or flows.
// CHECK-LABEL: @already_wired
module @already_wired {
  aie.device(npu3) {
    // CHECK: aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
    %shim = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
    %mem = aie.tile(0, 1)
    aie.objectfifo @in(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%mem, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // CHECK: aie.packet_flow(1) {
    aie.packet_flow(1) {
      aie.packet_source<%shim, Control32 : 0>
      aie.packet_dest<%shim, Controller32 : 0>
    }
    // Exactly one controller_id and one packet_flow remain.
    // CHECK-NOT: controller_id
    // CHECK-NOT: aie.packet_flow
  }
}

// -----

// A design that carries a controller_id but has NO Controller32 destination is
// not actually wired, so the wiring must still be emitted -- and it must reuse
// the controller_id that is already there rather than renumbering it, because
// the column control overlay routes control packets by those ids.
//
// This is the case the "is it already wired?" check gets wrong when it keys off
// the presence of a controller_id attribute instead of an existing TCT route.
// test/unit_tests/aie4_cert/10_iron_Nshim_1uc is a real design of this shape.
// CHECK-LABEL: @controller_id_but_no_tct_route
module @controller_id_but_no_tct_route {
  aie.device(npu3) {
    // CHECK: %[[SHIM:.*]] = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
    %shim = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
    %mem = aie.tile(0, 1)
    aie.objectfifo @in(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%mem, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // The existing pkt_id 7 is preserved and used as the flow ID.
    // CHECK: aie.packet_flow(7) {
    // CHECK-NEXT: aie.packet_source<%[[SHIM]], Control32 : 0>
    // CHECK-NEXT: aie.packet_dest<%[[SHIM]], Controller32 : 0>
    // CHECK: aie.packet_flow(7) {
    // CHECK-NEXT: aie.packet_source<%[[SHIM]], Control32 : 1>
    // CHECK-NEXT: aie.packet_dest<%[[SHIM]], Controller32 : 0>
  }
}

// -----

// Non-AIE4 targets (npu1) must not get any control wiring.
// CHECK-LABEL: @npu1_no_wiring
module @npu1_no_wiring {
  aie.device(npu1) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %m = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%m, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // CHECK-NOT: controller_id
    // CHECK-NOT: aie.packet_flow
  }
}

//===- aie4_switch32_512_bridge.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Regression test for routing across the AIE4 32-bit <-> 512-bit stream switch
// bridge, which had never worked end to end.
//
// An AIE4 tile holds two independent switches. The 32-bit switch's
// `512b_switch{k}` MANAGER port is hard-wired to the
// 512-bit switch's `32b_switch{k}` SUBORDINATE port, at matching index k. This
// dialect calls those ports Switch512 and Switch32 respectively, so crossing the
// bridge is two crossbar hops in one tile with a wire in between -- not a single
// crossbar hop, and not something isLegalTileConnection can express.
//
// The trace path needs exactly this: a shim's trace_s2mm is the 512-bit switch's
// DMA:1 manager port, while trace sources sit on the 32-bit switch.
//
// Two things must hold, and both are load-bearing:
//   1. the flow routes at all, emitting BOTH halves of the crossing;
//   2. the two halves use the SAME bridge index. The wires are index-matched, so
//      driving Switch512:0 while the 512-bit switch listens on Switch32:1 builds
//      a config that compiles clean and silently drops every packet.

// CHECK-LABEL: aie.device(npu3)

// Case 1: a shim-local crossing, Trace32 (32-bit switch) -> DMA:1 (trace_s2mm on
// the 512-bit switch). The bridge is crossed inside tile (0, 0).
// CHECK:      %[[shim0:.*]] = aie.tile(0, 0)
// CHECK:      aie.switchbox(%[[shim0]]) {
// CHECK-DAG:    aie.masterset(Switch512 : [[BR0:[0-9]]]
// CHECK-DAG:    aie.packet_rules(Trace32 : 0) {
// CHECK-DAG:    aie.packet_rules(Switch32 : [[BR0]]) {

// Case 2: a core tile trace egressing to a shim. Exactly one crossing is
// required, but it need not be at the destination. Equal-cost paths can cross
// at the memtile or the shim; the shortest-path heap currently selects the
// memtile, then reaches the shim on the wide network.
// CHECK:      %[[shim1:.*]] = aie.tile(1, 0)
// CHECK:      aie.switchbox(%[[shim1]]) {
// CHECK:        aie.masterset(DMA : 1,
// CHECK:        aie.packet_rules(North : {{[0-9]}}) {

// The core tile itself only touches the 32-bit switch: no bridge hop here.
// CHECK:      %[[core:.*]] = aie.tile(1, 2)
// CHECK:      aie.switchbox(%[[core]]) {
// CHECK-NOT:    Switch512
// CHECK-NOT:    Switch32
// CHECK:        aie.packet_rules(Trace32 : 0) {

// The memtile emits both index-matched halves of the required crossing.
// CHECK:      %[[mem:.*]] = aie.tile(1, 1)
// CHECK:      aie.switchbox(%[[mem]]) {
// CHECK-DAG:    aie.masterset(Switch512 : [[BR1:[0-9]]]
// CHECK-DAG:    aie.packet_rules(North32 : {{[0-9]}}) {
// CHECK-DAG:    aie.packet_rules(Switch32 : [[BR1]]) {

module {
  aie.device(npu3) {
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %core1 = aie.tile(1, 2)

    // Shim-local trace: crosses the bridge in place.
    aie.packet_flow(1) {
      aie.packet_source<%shim0, Trace32 : 0>
      aie.packet_dest<%shim0, DMA : 1>
    } {keep_pkt_header = true}

    // Core tile trace: down the column, with one bridge crossing.
    aie.packet_flow(2) {
      aie.packet_source<%core1, Trace32 : 0>
      aie.packet_dest<%shim1, DMA : 1>
    } {keep_pkt_header = true}
  }
}

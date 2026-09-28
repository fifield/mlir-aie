// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows --convert-aie-to-transaction %s -o /dev/null

// Trace ID 3 reaches both memory DMAs even when each output is assigned a
// different arbiter. It branches at the core: one branch crosses to the wide
// switch there, and the other crosses at memory. Each branch crosses once.
// CHECK: aie.switchbox(%mem_tile_0_1)
// CHECK: aie.masterset(DMA : 3, %[[DMA3:[a-zA-Z0-9_]+]])
// CHECK: aie.masterset(DMA : 6, %[[DMA6:[a-zA-Z0-9_]+]])
// CHECK: aie.masterset(Switch512 : [[MEMBR:[0-9]+]], %[[MEM_AMSEL:[a-zA-Z0-9_]+]])
// CHECK: aie.rule(31, 10,
// CHECK: aie.packet_rules(North : [[WIDE:[0-9]+]]) {
// CHECK-NEXT: aie.rule(31, 3, %[[DMA6]])
// CHECK: aie.packet_rules(North32 : [[NARROW:[0-9]+]]) {
// CHECK-NEXT: aie.rule(31, 3, %[[MEM_AMSEL]])
// CHECK: aie.packet_rules(Switch32 : [[MEMBR]]) {
// CHECK-NEXT: aie.rule(31, 3, %[[DMA3]])
// CHECK: aie.switchbox(%tile_0_2)
// CHECK: aie.masterset(South : [[WIDE]], %[[CORE_WIDE:[a-zA-Z0-9_]+]])
// CHECK: aie.masterset(South32 : [[NARROW]], %[[FANOUT:[a-zA-Z0-9_]+]])
// CHECK: aie.masterset(Switch512 : [[COREBR:[0-9]+]], %[[FANOUT]])
// CHECK: aie.packet_rules(Trace32 : 0) {
// CHECK-NEXT: aie.rule(31, 3, %[[FANOUT]])
// CHECK: aie.packet_rules(Switch32 : [[COREBR]]) {
// CHECK-NEXT: aie.rule(31, 3, %[[CORE_WIDE]])

module {
  aie.device(npu3_2col) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    %upper = aie.tile(0, 3)
    %shim1 = aie.tile(1, 0)
    aie.packet_flow(11) {
      aie.packet_source<%upper, DMA : 0>
      aie.packet_dest<%mem, DMA : 3>
    }
    aie.packet_flow(12) {
      aie.packet_source<%shim, Trace32 : 0>
      aie.packet_dest<%mem, DMA : 6>
    }
    aie.packet_flow(3) {
      aie.packet_source<%core, Trace32 : 0>
      aie.packet_dest<%mem, DMA : 3>
      aie.packet_dest<%mem, DMA : 6>
    }
    aie.packet_flow(10) {
      aie.packet_source<%core, DMA : 0>
      aie.packet_dest<%shim1, Controller32 : 0>
    }
  }
}

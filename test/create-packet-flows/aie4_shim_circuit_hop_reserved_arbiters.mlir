// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s --implicit-check-not=aie.shim_mux
// RUN: not aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 | FileCheck %s --check-prefix=NOHOPS
// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s --check-prefix=FLOW

// Every shim amsel is reserved. The direct AIE4 DMA input can still drive
// an exclusive northbound circuit hop: the packet is selected and its header
// dropped at the memtile. Legacy shim-mux assumptions must neither reject the
// source during the upfront arbiter check nor disable circuit-hop extraction.
// Disabling that optimization must still reject the exhausted shim.

// CHECK-LABEL: aie.switchbox(%shim_noc_tile_0_0)
// CHECK: aie.connect<DMA : 0, North : [[CHANNEL:[0-9]+]]>
// CHECK-NOT: aie.packet_rules(DMA : 0)
// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK: aie.masterset(DMA : 0, %[[A:.*]])
// CHECK: aie.packet_rules(South : [[CHANNEL]]) {
// CHECK-NEXT: aie.rule(31, 7, %[[A]])

// NOHOPS: error: Unable to find a legal routing: at tile (0, 0),
// NOHOPS-SAME: takes an arbiter whatever the routing, but the switchbox has none free

// FLOW: %[[SHIM:.*]] = aie.tile(0, 0)
// FLOW: %[[MEM:.*]] = aie.tile(0, 1)
// FLOW: aie.packet_flow(7) {
// FLOW-NEXT: aie.packet_source<%[[SHIM]], DMA : 0>
// FLOW-NEXT: aie.packet_dest<%[[MEM]], DMA : 0>
// FLOW-NEXT: }
// FLOW-NOT: aie.packet_flow

module {
  aie.device(npu3_1col) {
    %shim = aie.tile(0, 0)
    aie.switchbox(%shim) {
      %a00 = aie.amsel<0> (0)
      %a01 = aie.amsel<0> (1)
      %a02 = aie.amsel<0> (2)
      %a03 = aie.amsel<0> (3)
      %a10 = aie.amsel<1> (0)
      %a11 = aie.amsel<1> (1)
      %a12 = aie.amsel<1> (2)
      %a13 = aie.amsel<1> (3)
      %a20 = aie.amsel<2> (0)
      %a21 = aie.amsel<2> (1)
      %a22 = aie.amsel<2> (2)
      %a23 = aie.amsel<2> (3)
      %a30 = aie.amsel<3> (0)
      %a31 = aie.amsel<3> (1)
      %a32 = aie.amsel<3> (2)
      %a33 = aie.amsel<3> (3)
      %a40 = aie.amsel<4> (0)
      %a41 = aie.amsel<4> (1)
      %a42 = aie.amsel<4> (2)
      %a43 = aie.amsel<4> (3)
      %a50 = aie.amsel<5> (0)
      %a51 = aie.amsel<5> (1)
      %a52 = aie.amsel<5> (2)
      %a53 = aie.amsel<5> (3)
    }
    %mem = aie.tile(0, 1)
    aie.packet_flow(7) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%mem, DMA : 0>
    }
  }
}

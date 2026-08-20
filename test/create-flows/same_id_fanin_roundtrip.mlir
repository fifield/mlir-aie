// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s
//
// Preserve the original Medusa #258 fixture's end-to-end packet destinations
// across routing and lifting, independently of its chosen links and arbiters.
// Same-ID sources may merge, but neither may acquire another destination.
// CHECK-NOT: aie.packet_flow
// CHECK: aie.packet_flow(2) {
// CHECK-NEXT: aie.packet_source<%mem_tile_1_1, DMA : 5>
// CHECK-NEXT: aie.packet_dest<%tile_3_5, DMA : 0>
// CHECK-NEXT: }
// CHECK-NOT: aie.packet_flow
// CHECK: aie.packet_flow(0) {
// CHECK-NEXT: aie.packet_source<%tile_2_3, DMA : 0>
// CHECK-NEXT: aie.packet_dest<%mem_tile_1_1, DMA : 3>
// CHECK-NEXT: }
// CHECK-NOT: aie.packet_flow
// CHECK: aie.packet_flow(0) {
// CHECK-NEXT: aie.packet_source<%tile_2_5, DMA : 0>
// CHECK-NEXT: aie.packet_dest<%mem_tile_1_1, DMA : 3>
// CHECK-NEXT: }
// CHECK-NOT: aie.packet_flow

module {
  aie.device(npu2) {
    %mem_tile_1_1 = aie.tile(1, 1)
    %tile_2_3 = aie.tile(2, 3)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_5 = aie.tile(3, 5)
    aie.packet_flow(2) {
      aie.packet_source<%mem_tile_1_1, DMA : 5>
      aie.packet_dest<%tile_3_5, DMA : 0>
    }
    aie.packet_flow(0) {
      aie.packet_source<%tile_2_3, DMA : 0>
      aie.packet_dest<%mem_tile_1_1, DMA : 3>
    }
    aie.packet_flow(0) {
      aie.packet_source<%tile_2_5, DMA : 0>
      aie.packet_dest<%mem_tile_1_1, DMA : 3>
    }
  }
}

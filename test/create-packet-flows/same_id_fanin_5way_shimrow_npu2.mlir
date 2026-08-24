//===- same_id_fanin_5way_shimrow_npu2.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Widest same-id fan-in case, and the only one that merges progressively over
// several hops: five sources on the npu2 shim row funnel westward into a single
// mem_tile S2MM DMA channel. Companion to same_id_fanin_3way_npu1.mlir (npu1,
// single merge point) and same_id_shared_link.mlir (npu2, divergent form).
//
// It also carries a regression: the pathfinder must not SILENTLY drop a
// packet_source. This placement has a history:
//
//   - Originally the router compiled it "successfully" while delivering only
//     four of the five sources on device -- one source's connection was
//     dropped without a diagnostic.
//   - #3514 made that failure loud: an incomplete routing now reports
//     "could not be routed to destination".
//   - The reason it was unroutable at all was #3472's id-gated channel
//     sharing, which forbade two same-id flows from sharing a link, so each
//     of the five sources needed its own physical path. #258 relaxed that to
//     reject only fan-OUT (see same_id_fanin_3way_npu1.mlir), and same-id
//     fan-IN is exactly what this placement wants: the five flows now merge
//     progressively westward along the shim row and arrive as one stream.
//
// So the placement routes, and what this test pins is the original property
// stated positively: all five sources are routed, none silently dropped. Each
// source shim tile emits its own DMA (South : 3, through the shim mux) into
// the westward chain, and the mem_tile receives the merged stream on DMA : 0.
//
// TODO: the "fail loudly on incomplete routing" path from #3514 no longer has
// a lit trigger -- after #258 this input routes, and congestion-based attempts
// to rebuild an unroutable placement (distinct destinations, single-column
// devices, over-subscribing the 4-channel north links) all find legal
// routings. That error path is currently uncovered and wants a new test.

// CHECK-LABEL: aie.device(npu2)

// The merged stream lands on the mem_tile's DMA : 0.
// CHECK: aie.switchbox(%{{.*}}mem_tile_0_1)
// CHECK: aie.masterset(DMA : 0

// Every one of the five sources contributes its own DMA into the chain; a
// dropped source would leave its shim tile without a South : 3 rule.
// CHECK: aie.switchbox(%{{.*}}shim_noc_tile_2_0)
// CHECK: aie.packet_rules(South : 3)
// CHECK: aie.switchbox(%{{.*}}shim_noc_tile_3_0)
// CHECK: aie.packet_rules(South : 3)
// CHECK: aie.switchbox(%{{.*}}shim_noc_tile_4_0)
// CHECK: aie.packet_rules(South : 3)
// CHECK: aie.switchbox(%{{.*}}shim_noc_tile_5_0)
// CHECK: aie.packet_rules(South : 3)
// CHECK: aie.switchbox(%{{.*}}shim_noc_tile_6_0)
// CHECK: aie.packet_rules(South : 3)

// All five flows survive the pass.
// CHECK-COUNT-5: aie.packet_flow(0)

module {
  aie.device(npu2) {
    %mem_tile_0_1 = aie.tile(0, 1)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_2_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_3_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_4_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_5_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_6_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 0>
    } {keep_pkt_header = true}
  }
}

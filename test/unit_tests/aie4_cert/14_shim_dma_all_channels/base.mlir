//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices Inc.
//
// REQUIRES: dont_run
// RUN: echo FAIL | FileCheck %s
// CHECK: PASS

module {
  aie.device(npu3z) {
    // Shim tiles (col 0-8): tile(0,0) has elf_file, all have controller_ids
    %tile_0_0 = aie.tile(0, 0) { elf_file = "cert_medusa.elf", controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %tile_1_0 = aie.tile(1, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
    %tile_2_0 = aie.tile(2, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
    %tile_3_0 = aie.tile(3, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
    %tile_4_0 = aie.tile(4, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %tile_5_0 = aie.tile(5, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
    %tile_6_0 = aie.tile(6, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
    %tile_7_0 = aie.tile(7, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
    %tile_8_0 = aie.tile(8, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 8>}

    // Memtiles (col 0-7, row 1)
    %tile_0_1 = aie.tile(0, 1) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 9>}
    %tile_1_1 = aie.tile(1, 1) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 10>}
    %tile_2_1 = aie.tile(2, 1) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 11>}
    %tile_3_1 = aie.tile(3, 1) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 12>}
    %tile_4_1 = aie.tile(4, 1) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 13>}
    %tile_5_1 = aie.tile(5, 1) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 14>}
    %tile_6_1 = aie.tile(6, 1) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_7_1 = aie.tile(7, 1) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 16>}

    // Packet flows: route Control32 from all tiles to tile(0,0) Controller32:0
    // Shim tiles
    aie.packet_flow(0) {
      aie.packet_source<%tile_0_0, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(1) {
      aie.packet_source<%tile_1_0, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(2) {
      aie.packet_source<%tile_2_0, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(3) {
      aie.packet_source<%tile_3_0, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(4) {
      aie.packet_source<%tile_4_0, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(5) {
      aie.packet_source<%tile_5_0, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(6) {
      aie.packet_source<%tile_6_0, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(7) {
      aie.packet_source<%tile_7_0, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(8) {
      aie.packet_source<%tile_8_0, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    // Memtiles
    aie.packet_flow(9) {
      aie.packet_source<%tile_0_1, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(10) {
      aie.packet_source<%tile_1_1, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(11) {
      aie.packet_source<%tile_2_1, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(12) {
      aie.packet_source<%tile_3_1, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(13) {
      aie.packet_source<%tile_4_1, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(14) {
      aie.packet_source<%tile_5_1, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%tile_6_1, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(16) {
      aie.packet_source<%tile_7_1, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}

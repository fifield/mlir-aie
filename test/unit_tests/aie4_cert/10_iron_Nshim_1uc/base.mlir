//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
// REQUIRES: mesimulator
//
// RUN: %python %S/vector_vector_mul.py --output aie4.mlir
// RUN: mkdir -p project0/
// RUN: cp %aie_obj_root/third_party/cert/main/cert_medusa.elf project0/cert_medusa.elf
// RUN: aiecc.py --tmpdir=project0 --xchesscc --xbridge --aie-generate-txn --txn-name=transaction.bin %s
// RUN: aiecc.py --tmpdir=project1 --xchesscc --xbridge --no-compile-host --aie-generate-elf --elf-name=aie4.elf --aiesim aie4.mlir %test_utils_flags %S/test.cpp
// RUN: project1/mesim.sh | FileCheck --check-prefix=MESIM %s
//
// MESIM: PASS!
//
module {
  aie.device(npu3z) {
    %tile_0_0 = aie.tile(0, 0) { elf_file = "cert_medusa.elf", controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %tile_1_0 = aie.tile(1, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
    %tile_2_0 = aie.tile(2, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
    %tile_3_0 = aie.tile(3, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
    %tile_4_0 = aie.tile(4, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %tile_5_0 = aie.tile(5, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
    %tile_6_0 = aie.tile(6, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
    %tile_7_0 = aie.tile(7, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
    %tile_8_0 = aie.tile(8, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 8>}
    // Route from control_rsp port 0 of shims to ucontroller port 0 of shim (0,0) for TCT packets
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
  }
}

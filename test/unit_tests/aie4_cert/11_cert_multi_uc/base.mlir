//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
// REQUIRES: mesimulator
//
// RUN: mkdir -p project0/
// RUN: cp %aie_obj_root/third_party/cert/main/cert_medusa.elf project0/cert_medusa.elf
// RUN: aiecc --tmpdir=project0 --xchesscc --xbridge --aie-generate-txn --txn-name=transaction.bin %s
// RUN: aiecc --tmpdir=project1 --xchesscc --xbridge --no-compile-host --get-full-elf --full-elf-name=aie4.elf --aiesim %S/aie4.mlir %test_utils_flags %S/test.cpp
// RUN: project1/mesim.sh | FileCheck --check-prefix=MESIM %s
//
// MESIM: device model: {{.*}}aie4
// MESIM: PASS!
//
// segfaults in CI, works running manually.
// XFAIL: *

module {
  aie.device(npu3) {
    %tile_0_0 = aie.tile(0, 0) { elf_file = "cert_medusa.elf", controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
    %tile_1_0 = aie.tile(1, 0) { elf_file = "cert_medusa.elf", controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_0, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(2) {
      aie.packet_source<%tile_1_0, Control32 : 0>
      aie.packet_dest<%tile_1_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}

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
  aie.device(npu3) {
    %tile_0_0 = aie.tile(0, 0) { controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>,elf_file = "cert_medusa.elf" }

    aie.packet_flow(4) {
      aie.packet_source<%tile_0_0, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}

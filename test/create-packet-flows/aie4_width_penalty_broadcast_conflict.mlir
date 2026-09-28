// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: not aie-opt --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s

// Local switch inputs force direct connections, independent of equal-cost
// routing detours elsewhere in the array. Earlier unicasts assign DMA3 and
// DMA6 to different arbiters. A single bridge input cannot broadcast to both
// arbiters; reject the conflict instead of dropping one destination.
// CHECK: cannot allocate packet broadcast: output ports are already assigned to different arbiters

module {
  aie.device(npu3_1col) {
    %mem = aie.tile(0, 1)
    aie.packet_flow(11) {
      aie.packet_source<%mem, North : 1>
      aie.packet_dest<%mem, DMA : 3>
    }
    aie.packet_flow(12) {
      aie.packet_source<%mem, North : 2>
      aie.packet_dest<%mem, DMA : 6>
    }
    aie.packet_flow(3) {
      aie.packet_source<%mem, Switch32 : 0>
      aie.packet_dest<%mem, DMA : 3>
      aie.packet_dest<%mem, DMA : 6>
    }
  }
}

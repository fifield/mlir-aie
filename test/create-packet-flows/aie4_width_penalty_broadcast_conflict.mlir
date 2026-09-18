// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: not aie-opt --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s

// The width preference exposes a latent greedy multicast-allocation conflict.
// ID 3 takes a single bridge crossing then broadcasts to DMA:3 and DMA:6.
// Earlier unicasts pin these outputs to different arbiters. Silently skipping
// DMA:6 used to emit a successful configuration that dropped half of ID 3.
// Until allocation handles multicast output components globally, reject the
// conflict rather than generating incorrect hardware configuration.
// CHECK: cannot allocate packet broadcast: output ports are already assigned to different arbiters

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

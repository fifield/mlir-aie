//===- npu3_pinned_memtile_bd_pools.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Each npu3 memtile channel has a private 16-BD pool; a pinned runtime BD is
// charged to its own channel's pool only.

// RUN: aie-opt --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s

// BD 0 is pinned on S2MM 1, so S2MM 0 keeps all 16 of its BDs for @fa.
// CHECK-LABEL: @pinned_other_channel
// CHECK: aie.objectfifo.dma_endpoint @fa__cons__dma(%{{.*}}) fills @fa__cons__pool {channelIndex = 0 : i32
module @pinned_other_channel {
  aie.device(npu3) {
    %a = aie.tile(0, 2)
    %m = aie.tile(0, 1)
    %buf = aie.buffer(%m) {address = 0 : i32} : memref<16xi32>
    aie.runtime_sequence() {
      %task = aiex.dma_configure_task(%m, S2MM, 1) {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.end
      }
    }
    aie.objectfifo @fa(%a, {%m}, [2, 16]) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>, out_of_order = true>, iter_count = 1 : i32, cons_ports = [#aie.end_port<DMA : 0>]} : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @fa(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @fa(Produce, 1)
      aie.end
    }
  }
}

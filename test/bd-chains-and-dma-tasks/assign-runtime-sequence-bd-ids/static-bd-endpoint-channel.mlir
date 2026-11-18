//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// A static BD on a not-yet-allocated endpoint channel takes its id in every
// pool that holds it.

// CHECK: aie.dma_start(S2MM, @in,
// CHECK: aie.dma_bd({{.*}}) {bd_id = 0 : i32}
// CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 1 : i32

module {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    %buf = aie.buffer(%mem) {address = 0 : i32} : memref<1024xi32>
    aie.route_endpoint @src(%core) Core {channelIndex = 0 : i32}
    aie.route_endpoint @in(%mem) DMA
    aie.route from @src to [@in]
    %memdma = aie.memtile_dma(%mem) {
      %0 = aie.dma_start(S2MM, @in, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256) {bd_id = 0 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.runtime_sequence @seq() {
      %t0 = aiex.dma_configure_task(%mem, MM2S, 0) {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%t0)
      aiex.dma_await_task(%t0)
    }
  }
}

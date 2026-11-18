//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// AIE4 runtime BD pools: shim S2MM 1 uses the upper bank (16-31), MM2S 1 the
// lower; memtile channels have private pools with channel-local ids.

// CHECK-LABEL: @shim_banks
// CHECK: aiex.dma_configure_task(%{{.*}}, S2MM, 1)
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 16 : i32
// CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 1)
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 0 : i32
// CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 2)
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 17 : i32
// CHECK: aiex.dma_configure_task(%{{.*}}, S2MM, 0)
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 1 : i32

// CHECK-LABEL: @memtile_pools
// CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 0 : i32
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 1 : i32
// CHECK: aiex.dma_configure_task(%{{.*}}, S2MM, 0)
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 0 : i32
// CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 5)
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 0 : i32
// After an await, the ids are free again in that pool only.
// CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 0 : i32

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %buf = aie.buffer(%mem) : memref<1024xi32>
    aie.runtime_sequence @shim_banks(%arg0 : memref<1024xi32>) {
      %t0 = aiex.dma_configure_task(%shim, S2MM, 1) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      %t1 = aiex.dma_configure_task(%shim, MM2S, 1) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      %t2 = aiex.dma_configure_task(%shim, MM2S, 2) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      %t3 = aiex.dma_configure_task(%shim, S2MM, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
    }
    aie.runtime_sequence @memtile_pools() {
      %t0 = aiex.dma_configure_task(%mem, MM2S, 0) {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256)
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%buf : memref<1024xi32> offset = 256 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task(%mem, S2MM, 0) {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      %t2 = aiex.dma_configure_task(%mem, MM2S, 5) {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_await_task(%t0)
      %t3 = aiex.dma_configure_task(%mem, MM2S, 0) {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
    }
  }
}

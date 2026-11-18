//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// On AIE4 a static BD id is taken only in its own channel's pool, so other
// memtile channels and the shim control/trace pools can reuse it.

// CHECK-LABEL: @memtile
// CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 0 : i32
// CHECK: aiex.dma_configure_task(%{{.*}}, S2MM, 1)
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 0 : i32
// CHECK: aiex.dma_configure_task(%{{.*}}, S2MM, 0)
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 2 : i32

// CHECK-LABEL: @shim
// CHECK: aiex.dma_configure_task(%{{.*}}, ControlMM2S, 0)
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 0 : i32
// CHECK: aiex.dma_configure_task(%{{.*}}, TraceS2MM, 0)
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 0 : i32
// CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 1)
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 1 : i32

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %buf = aie.buffer(%mem) {address = 0 : i32} : memref<1024xi32>
    %ext = aie.external_buffer {sym_name = "ext"} : memref<1024xi32>
    %memdma = aie.memtile_dma(%mem) {
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%buf : memref<1024xi32> offset = 256 len = 256) {bd_id = 1 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %shimdma = aie.shim_dma(%shim) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%ext : memref<1024xi32> offset = 0 len = 256) {bd_id = 0 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.runtime_sequence @memtile() {
      %t0 = aiex.dma_configure_task(%mem, MM2S, 0) {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      %t1 = aiex.dma_configure_task(%mem, S2MM, 1) {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      %t2 = aiex.dma_configure_task(%mem, S2MM, 0) {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
    }
    aie.runtime_sequence @shim(%arg0 : memref<1024xi32>) {
      %t0 = aiex.dma_configure_task(%shim, ControlMM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      %t1 = aiex.dma_configure_task(%shim, TraceS2MM, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      %t2 = aiex.dma_configure_task(%shim, MM2S, 1) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
    }
  }
}

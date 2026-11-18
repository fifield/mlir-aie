//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// AIE4 shim ControlMM2S (0-7 each) and TraceS2MM (0-3) have private BD pools.

// CHECK-LABEL: @control_trace
// CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 0)
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 0 : i32
// CHECK: aiex.dma_configure_task(%{{.*}}, ControlMM2S, 0)
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 0 : i32
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 1 : i32
// CHECK: aiex.dma_configure_task(%{{.*}}, ControlMM2S, 1)
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 0 : i32
// CHECK: aiex.dma_configure_task(%{{.*}}, TraceS2MM, 0)
// CHECK:   aie.dma_bd({{.*}}) {bd_id = 0 : i32

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.runtime_sequence @control_trace(%arg0 : memref<1024xi32>) {
      %t0 = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      %t1 = aiex.dma_configure_task(%shim, ControlMM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 256 len = 256)
        aie.end
      }
      %t2 = aiex.dma_configure_task(%shim, ControlMM2S, 1) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      %t3 = aiex.dma_configure_task(%shim, TraceS2MM, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
    }
  }
}

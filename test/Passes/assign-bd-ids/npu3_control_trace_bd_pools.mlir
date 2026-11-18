//===- npu3_control_trace_bd_pools.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids %s | FileCheck %s

// AIE4 shim: each ControlMM2S channel has a private 8-BD pool and TraceS2MM a
// 4-BD pool, separate from the data DMA banks.

// CHECK-LABEL: aie.device(npu3)
// CHECK:       aie.dma_start(ControlMM2S, 0, ^[[C0:.*]], ^{{.*}})
// CHECK:       ^[[C0]]:
// CHECK:         aie.dma_bd({{.*}}) {bd_id = 0 : i32, next_bd_id = 1 : i32}
// CHECK:         aie.dma_bd({{.*}}) {bd_id = 1 : i32, next_bd_id = 0 : i32}
// CHECK:       aie.dma_start(ControlMM2S, 1, ^[[C1:.*]], ^{{.*}})
// CHECK:       ^[[C1]]:
// CHECK:         aie.dma_bd({{.*}}) {bd_id = 7 : i32, next_bd_id = 0 : i32}
// CHECK:         aie.dma_bd({{.*}}) {bd_id = 0 : i32}
// CHECK:       aie.dma_start(TraceS2MM, 0, ^[[T0:.*]], ^{{.*}})
// CHECK:       ^[[T0]]:
// CHECK:         aie.dma_bd({{.*}}) {bd_id = 0 : i32, next_bd_id = 0 : i32}
// CHECK:       aie.dma_start(MM2S, 0, ^[[D0:.*]], ^{{.*}})
// CHECK:       ^[[D0]]:
// CHECK:         aie.dma_bd({{.*}}) {bd_id = 0 : i32, next_bd_id = 0 : i32}
module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    %ext = aie.external_buffer : memref<256xi32>
    aie.shim_dma(%shim) {
      %0 = aie.dma_start(ControlMM2S, 0, ^c0a, ^c1)
    ^c0a:
      aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^c0b
    ^c0b:
      aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^c0a
    ^c1:
      %1 = aie.dma_start(ControlMM2S, 1, ^c1a, ^t0)
    ^c1a:
      aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256) {bd_id = 7 : i32}
      aie.next_bd ^c1b
    ^c1b:
      aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^end
    ^t0:
      %2 = aie.dma_start(TraceS2MM, 0, ^t0a, ^d0)
    ^t0a:
      aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^t0a
    ^d0:
      %3 = aie.dma_start(MM2S, 0, ^d0a, ^end)
    ^d0a:
      aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^d0a
    ^end:
      aie.end
    }
  }
}

//===- npu3_control_trace_dma.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// AIE4 shim ControlMM2S (2) and TraceS2MM (1) channels verify and round-trip.

// RUN: aie-opt %s | FileCheck %s
// RUN: aie-opt %s | aie-opt | FileCheck %s

// CHECK-LABEL: aie.device(npu3)
// CHECK:       aie.shim_dma_allocation @ctrl0(%{{.*}}, ControlMM2S, 0)
// CHECK:       aie.shim_dma_allocation @ctrl1(%{{.*}}, ControlMM2S, 1)
// CHECK:       aie.shim_dma_allocation @trace0(%{{.*}}, TraceS2MM, 0)
// CHECK:       aie.shim_dma
// CHECK:         aie.dma_start(ControlMM2S, 0,
// CHECK:         aie.dma_start(ControlMM2S, 1,
// CHECK:         aie.dma_start(TraceS2MM, 0,
// CHECK:         aie.dma_start(MM2S, 0,
// CHECK:       aie.runtime_sequence
// CHECK:         aiex.dma_configure_task(%{{.*}}, ControlMM2S, 1)
// CHECK:         aiex.dma_configure_task(%{{.*}}, TraceS2MM, 0)
// CHECK:         aiex.npu.push_queue(1, 0, ControlMM2S : 0)
// CHECK:         aiex.npu.push_queue(1, 0, TraceS2MM : 0)
// CHECK:         aiex.dma_start_bd_chain @chain({{.*}}) on (%{{.*}}, ControlMM2S, 0)
module {
  aie.device(npu3) {
    %shim = aie.tile(1, 0)
    %ext = aie.external_buffer : memref<64xi32>
    aie.shim_dma_allocation @ctrl0(%shim, ControlMM2S, 0)
    aie.shim_dma_allocation @ctrl1(%shim, ControlMM2S, 1)
    aie.shim_dma_allocation @trace0(%shim, TraceS2MM, 0)
    aie.shim_dma(%shim) {
      %0 = aie.dma_start(ControlMM2S, 0, ^c0, ^c1start)
    ^c0:
      aie.dma_bd(%ext : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^end
    ^c1start:
      %1 = aie.dma_start(ControlMM2S, 1, ^c1, ^tstart)
    ^c1:
      aie.dma_bd(%ext : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^end
    ^tstart:
      %2 = aie.dma_start(TraceS2MM, 0, ^t0, ^dstart)
    ^t0:
      aie.dma_bd(%ext : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^end
    ^dstart:
      %3 = aie.dma_start(MM2S, 0, ^d0, ^end)
    ^d0:
      aie.dma_bd(%ext : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.bd_chain @chain(%buf: memref<64xi32>) {
      aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64)
      aie.end
    }
    aie.runtime_sequence @seq(%arg0 : memref<64xi32>) {
      %t0 = aiex.dma_configure_task(%shim, ControlMM2S, 1) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      %t1 = aiex.dma_configure_task(%shim, TraceS2MM, 0) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      %c0 = arith.constant 0 : i32
      aiex.npu.push_queue (1, 0, ControlMM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (1, 0, TraceS2MM:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
      %t2 = aiex.dma_start_bd_chain @chain(%arg0) : (memref<64xi32>) on (%shim, ControlMM2S, 0)
    }
  }
}

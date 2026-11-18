//===- bad_trace_s2mm_bd_pool_aie4.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --canonicalize %s 2>&1 | FileCheck %s
// CHECK: BD pool for channel (TraceS2MM, 0) exceeds limit of 4 BDs

// AIE4 shim trace-S2MM has a private 4-BD pool, not the 16-BD S2MM pool.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<256xi32>
  %dma = aie.shim_dma(%shim) {
    %start = aie.dma_start("TraceS2MM", 0, ^a0, ^end)
  ^a0:
    aie.dma_bd(%buffer : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^a1
  ^a1:
    aie.dma_bd(%buffer : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^a2
  ^a2:
    aie.dma_bd(%buffer : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^a3
  ^a3:
    aie.dma_bd(%buffer : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^a4
  ^a4:
    aie.dma_bd(%buffer : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^a4
  ^end:
    aie.end
  }
}

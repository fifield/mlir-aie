//===- npu3_control_trace_bd_pools_errors.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids --verify-diagnostics --split-input-file %s

// AIE4 shim control and trace BD pools: a pinned id must lie in the channel's
// pool, and a pool holds 8 (ControlMM2S) or 4 (TraceS2MM) BDs.

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %ext = aie.external_buffer : memref<256xi32>
  aie.shim_dma(%shim) {
    %0 = aie.dma_start(ControlMM2S, 1, ^bd0, ^end)
  ^bd0:
    // expected-error@+1 {{assigned bd_id 8 is not accessible from channel 1 on this tile}}
    aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256) {bd_id = 8 : i32}
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %ext = aie.external_buffer : memref<256xi32>
  aie.shim_dma(%shim) {
    %0 = aie.dma_start(TraceS2MM, 0, ^bd0, ^end)
  ^bd0:
    aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd1
  ^bd1:
    aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd2
  ^bd2:
    aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd3
  ^bd3:
    aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd4
  ^bd4:
    // expected-error@+1 {{Allocator exhausted available BD IDs (maximum 4 available for channel 0)}}
    aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd0
  ^end:
    aie.end
  }
}

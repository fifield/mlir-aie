//===- npu3_control_trace_bd_pools_errors.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids --verify-diagnostics --split-input-file %s

// AIE4 shim ControlMM2S (8 BDs) and TraceS2MM (4 BDs) pool limits.

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %ext = aie.external_buffer : memref<256xi32>
  // expected-error@+1 {{bd_id 8 is not accessible from channel (ControlMM2S, 1), whose BD pool spans ids 0 to 7}}
  aie.shim_dma(%shim) {
    %0 = aie.dma_start(ControlMM2S, 1, ^bd0, ^end)
  ^bd0:
    // expected-note@+1 {{this BD}}
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
  // expected-error@+1 {{BD pool for channel (TraceS2MM, 0) exceeds limit of 4 BDs}}
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
    // expected-note@+1 {{no space for this BD}}
    aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd0
  ^end:
    aie.end
  }
}

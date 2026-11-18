//===- shim_mixed_bd_pools_aie4.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --canonicalize %s | FileCheck %s
// CHECK: aie.device

// AIE4 shim: 32 bank BDs plus one from the private 4-BD TraceS2MM pool fit.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<64xi32>
  %dma = aie.shim_dma(%shim) {
    %s2mm = aie.dma_start(S2MM, 0, ^a0, ^mm2s_start)
  ^a0:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^a1
  ^a1:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^a2
  ^a2:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^a3
  ^a3:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^a4
  ^a4:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^a5
  ^a5:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^a6
  ^a6:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^a7
  ^a7:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^a8
  ^a8:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^a9
  ^a9:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^a10
  ^a10:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^a11
  ^a11:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^a12
  ^a12:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^a13
  ^a13:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^a14
  ^a14:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^a15
  ^a15:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^a15
  ^mm2s_start:
    %mm2s = aie.dma_start(MM2S, 2, ^b0, ^trace_start)
  ^b0:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^b1
  ^b1:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^b2
  ^b2:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^b3
  ^b3:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^b4
  ^b4:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^b5
  ^b5:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^b6
  ^b6:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^b7
  ^b7:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^b8
  ^b8:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^b9
  ^b9:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^b10
  ^b10:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^b11
  ^b11:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^b12
  ^b12:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^b13
  ^b13:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^b14
  ^b14:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^b15
  ^b15:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^b15
  ^trace_start:
    %trace = aie.dma_start(TraceS2MM, 0, ^trace_bd, ^end)
  ^trace_bd:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^trace_bd
  ^end:
    aie.end
  }
}

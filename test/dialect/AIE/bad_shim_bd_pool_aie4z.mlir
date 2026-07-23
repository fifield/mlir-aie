//===- bad_shim_bd_pool_aie4z.mlir -*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --canonicalize %s 2>&1 | FileCheck %s
// CHECK: BD pool for channel (MM2S, 0) exceeds limit of 16 BDs

// AIE4z shim: each channel has 16 private BDs. This uses 17 = overflow.
aie.device(npu3z) {
  %t1 = aie.tile(0, 0)
  %buf = aie.external_buffer : memref<256xi32>
  %mem = aie.shim_dma(%t1) {
    %dma0 = aie.dma_start("MM2S", 0, ^a0, ^end)
    ^a0:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^a1
    ^a1:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^a2
    ^a2:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^a3
    ^a3:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^a4
    ^a4:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^a5
    ^a5:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^a6
    ^a6:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^a7
    ^a7:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^a8
    ^a8:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^a9
    ^a9:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^a10
    ^a10:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^a11
    ^a11:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^a12
    ^a12:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^a13
    ^a13:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^a14
    ^a14:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^a15
    ^a15:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^a16
    ^a16:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^a16
    ^end:
      aie.end
  }
}

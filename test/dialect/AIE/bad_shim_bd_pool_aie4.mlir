//===- bad_shim_bd_pool_aie4.mlir -*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --canonicalize %s 2>&1 | FileCheck %s
// CHECK: BD pool for channel (MM2S, 1) exceeds limit of 16 BDs

// AIE4 shim: BDs 0-15 shared by MM2S channels 0 and 1.
// 9 BDs on ch0 + 8 BDs on ch1 = 17 > 16 pool limit.
aie.device(npu3) {
  %t1 = aie.tile(0, 0)
  %buf = aie.external_buffer : memref<256xi32>
  %mem = aie.shim_dma(%t1) {
    %dma0 = aie.dma_start("MM2S", 0, ^a0, ^ch1_start)
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
      aie.next_bd ^a8
    ^ch1_start:
      %dma1 = aie.dma_start("MM2S", 1, ^b0, ^end)
    ^b0:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^b1
    ^b1:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^b2
    ^b2:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^b3
    ^b3:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^b4
    ^b4:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^b5
    ^b5:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^b6
    ^b6:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^b7
    ^b7:
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^b7
    ^end:
      aie.end
  }
}

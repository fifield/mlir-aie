//===- bad_memtile_bd_pool_shared_aie2.mlir -*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --canonicalize %s 2>&1 | FileCheck %s
// CHECK: BD pool for channel (MM2S, 2) exceeds limit of 24 BDs

// AIE2 memtile: even channels (0 and 2) share BDs 0-23.
// 12 BDs on ch0 + 13 BDs on ch2 = 25 total from a pool of 24.
aie.device(npu1) {
  %t1 = aie.tile(0, 1)
  %buf = aie.buffer(%t1) : memref<256xi32>
  %mem = aie.memtile_dma(%t1) {
    %dma0 = aie.dma_start("MM2S", 0, ^a0, ^ch1_start)
    ^a0:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a1
    ^a1:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a2
    ^a2:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a3
    ^a3:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a4
    ^a4:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a5
    ^a5:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a6
    ^a6:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a7
    ^a7:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a8
    ^a8:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a9
    ^a9:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a10
    ^a10:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a11
    ^a11:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a11
    ^ch1_start:
      %dma1 = aie.dma_start("MM2S", 2, ^b0, ^end)
    ^b0:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^b1
    ^b1:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^b2
    ^b2:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^b3
    ^b3:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^b4
    ^b4:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^b5
    ^b5:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^b6
    ^b6:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^b7
    ^b7:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^b8
    ^b8:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^b9
    ^b9:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^b10
    ^b10:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^b11
    ^b11:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^b12
    ^b12:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^b12
    ^end:
      aie.end
  }
}

//===- memtile_bd_pool_limit_aie2.mlir -*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --canonicalize %s | FileCheck %s
// CHECK: aie.device

// AIE2 memtile: even channels share BDs 0-23 (24 BDs). This uses exactly 24.
aie.device(npu1) {
  %t1 = aie.tile(0, 1)
  %buf = aie.buffer(%t1) : memref<256xi32>
  %mem = aie.memtile_dma(%t1) {
    %dma0 = aie.dma_start("MM2S", 0, ^a0, ^end)
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
      aie.next_bd ^a12
    ^a12:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a13
    ^a13:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a14
    ^a14:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a15
    ^a15:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a16
    ^a16:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a17
    ^a17:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a18
    ^a18:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a19
    ^a19:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a20
    ^a20:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a21
    ^a21:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a22
    ^a22:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a23
    ^a23:
      aie.dma_bd(%buf : memref<256xi32>, 0, 256)
      aie.next_bd ^a23
    ^end:
      aie.end
  }
}

//===- aie2_tileDMA.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: XAie_DmaDesc [[bd0:.*]];
// CHECK: XAie_DmaDescInit(&(ctx->DevInst), &([[bd0]]), XAie_TileLoc(7,4))
// CHECK: XAie_DmaSetLock(&([[bd0]]), XAie_LockInit(3,-1),XAie_LockInit(4,1))
// CHECK: XAie_DmaSetAddrLen(&([[bd0]]),  /* addrA */ 0x720,  /* len */ 256 * 4)
// CHECK: XAie_DmaSetNextBd(&([[bd0]]),  /* nextbd */ 1,  /* enableNextBd */ 1)
// CHECK: XAie_DmaEnableBd(&([[bd0]]))
// CHECK: XAie_DmaWriteBd(&(ctx->DevInst), &([[bd0]]), XAie_TileLoc(7,4),  /* bd */ 0)
// CHECK: XAie_DmaChannelPushBdToQueue(&(ctx->DevInst), XAie_TileLoc(7,4), /* ChNum */0, /* dmaDir */ DMA_S2MM, /* BdNum */0)
// CHECK: XAie_DmaChannelEnable(&(ctx->DevInst), XAie_TileLoc(7,4), /* ChNum */ 0, /* dmaDir */ DMA_S2MM)

module @aie_module  {
  AIE.device(xcve2802) {
    %t63 = AIE.tile(6, 4)
    %t73 = AIE.tile(7, 4)
    %t72 = AIE.tile(7, 3)
    %t74 = AIE.tile(7, 5)

    %buf_e = AIE.buffer(%t63) {address = 0 : i32, sym_name = "east" } : memref<256xi32>
    %buf_l = AIE.buffer(%t73) {address = 1824 : i32, sym_name = "local" } : memref<256xi32>
    %buf_n = AIE.buffer(%t74) {address = 0 : i32, sym_name = "north" } : memref<256xi32>
    %buf_s = AIE.buffer(%t72) {address = 0 : i32, sym_name = "south" } : memref<256xi32>

    %lock_e = AIE.lock(%t63, 0)
    %lock_l1 = AIE.lock(%t73, 3)
    %lock_l2 = AIE.lock(%t73, 4)
    %lock_n = AIE.lock(%t74, 0)
    %lock_s = AIE.lock(%t72, 0)
    
    // Tile DMA
    %m73 = AIE.mem(%t73) {
        %srcDma = AIE.dmaStart("S2MM", 0, ^bd0, ^end)
      ^bd0:
        AIE.useLock(%lock_l1, AcquireGreaterEqual, 1)
        AIE.dmaBd(<%buf_l : memref<256xi32>, 0, 256>, 0)
        AIE.useLock(%lock_l2, Release, 1)
        AIE.nextBd ^bd1
      ^bd1:
        AIE.dmaBd(<%buf_l : memref<256xi32>, 0, 256>, 0)
        AIE.useLock(%lock_l1, Release, 1)
        AIE.nextBd ^bd2
      ^bd2:
        AIE.dmaBd(<%buf_l : memref<256xi32>, 0, 256>, 0)
        AIE.useLock(%lock_l1, Release, 1)
        AIE.nextBd ^bd3
      ^bd3:
        AIE.dmaBd(<%buf_l : memref<256xi32>, 0, 256>, 0)
        AIE.useLock(%lock_l1, Release, 1)
        AIE.nextBd ^end
      ^end:
        AIE.end
    }
 }
}
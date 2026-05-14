//===- dma_to_npu_aie4_shim_shared_bd_blocks.mlir ---------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Verify that when block merging causes multiple DMA channels to share the
// same BD blocks on an npu3z shim tile, each channel's private BD pool is
// configured separately. This is a regression test for a bug where only the
// first channel's BDs were written.

// RUN: aie-opt -convert-aie-to-transaction %s | FileCheck %s

// Two MM2S channels (0 and 2) share the same BD block (^bb1).
// We expect blockwrites to both channel 0's and channel 2's private BD pools,
// plus the S2MM channel 0 BD, for a total of 3 blockwrites.
// We also expect task queue writes for all 3 channels.

// BD for MM2S ch0 private pool (bd_id=0)
// CHECK-LABEL: aie.runtime_sequence @configure
// CHECK: aiex.npu.blockwrite({{.*}}) {address = 101440 : ui32} : memref<9xi32>
// BD for MM2S ch2 private pool (bd_id=0) - same content, different address
// CHECK: aiex.npu.blockwrite({{.*}}) {address = 102976 : ui32} : memref<9xi32>
// BD for S2MM ch0 private pool (bd_id=0)
// CHECK: aiex.npu.blockwrite({{.*}}) {address = 98304 : ui32} : memref<9xi32>
// Task queue writes for MM2S ch0, MM2S ch2, and S2MM ch0
// CHECK: aiex.npu.write32 {address = 99668 : ui32
// CHECK: aiex.npu.write32 {address = 361812 : ui32
// CHECK: aiex.npu.write32 {address = 99652 : ui32

module {
  aie.device(npu3z) {
    %ext_buf = aie.external_buffer {address = 2147483648 : i64} : memref<64xi32>
    %tile_0_0 = aie.tile(0, 0)
    %lock_0 = aie.lock(%tile_0_0, 0) {init = 0 : i32}
    %lock_1 = aie.lock(%tile_0_0, 1) {init = 0 : i32}
    %shim_dma_0_0 = aie.shim_dma(%tile_0_0) {
      // MM2S channel 0 -> ^bb1
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // shared by both MM2S ch0 and ch2
      aie.use_lock(%lock_0, Acquire, 1)
      aie.dma_bd(%ext_buf : memref<64xi32>, 0, 64, [<size = 64, stride = 1>]) {bd_id = 0 : i32}
      aie.use_lock(%lock_1, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // end block
      aie.end
    ^bb3:
      // MM2S channel 2 -> ^bb1 (same block as channel 0!)
      %1 = aie.dma_start(MM2S, 2, ^bb1, ^bb4)
    ^bb4:
      // S2MM channel 0 -> ^bb5
      %2 = aie.dma_start(S2MM, 0, ^bb5, ^bb2)
    ^bb5:
      aie.use_lock(%lock_1, Acquire, 1)
      aie.dma_bd(%ext_buf : memref<64xi32>, 0, 64, [<size = 64, stride = 1>]) {bd_id = 0 : i32}
      aie.use_lock(%lock_0, Release, 1)
      aie.next_bd ^bb2
    }
  }
}

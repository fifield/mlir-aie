//===- npu3_memtile_core_bd_layout.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-dma-tasks-to-npu --aie-dma-to-npu --canonicalize %s | FileCheck %s

// AIE4 memtile and core BD packing; expected words match aie-codegen. The
// address word is written with other fields cleared, then masked in.
// A memtile DMA reaches memtiles up to 7 columns away via window
// w = column delta + 7: buffer 0xB00000 + w * 0x300000 + addr, lock w * 64 + id.

// Memtile (1, 1) MM2S 0 BD 2 at 0x02108C60; west buffer
// (0xB00000 + 6 * 0x300000 + 4096) / 4 = 0x740400; locks west 6 * 64 + 5 = 389,
// east 8 * 64 + 9 = 521.
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<11xi32> = dense<[0, -33553920, 407899155, 32, 256, 0, 0, 0, 0, 4194816, 0]>
// CHECK-LABEL: @memtile_west
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 34638944 : ui32} : memref<11xi32>
// CHECK: aiex.npu.maskwrite32(%c34638944_i32, %c7603200_i32, %c16777215_i32)

aie.device(npu3) {
  %m01 = aie.tile(0, 1)
  %m11 = aie.tile(1, 1)
  %m21 = aie.tile(2, 1)
  %west = aie.buffer(%m01) {address = 4096 : i32} : memref<1024xi32>
  %lw = aie.lock(%m01, 5)
  %le = aie.lock(%m21, 9)
  aie.runtime_sequence @memtile_west() {
    %c1 = arith.constant 1 : i32
    %task = aiex.dma_configure_task(%m11, MM2S, 0) {
      aie.use_lock(%lw, AcquireGreaterEqual, %c1)
      aie.dma_bd(%west : memref<1024xi32> offset = 0 len = 512 sizes = [1, 4, 8, 16] strides = [0, 256, 32, 1]) {bd_id = 2 : i32}
      aie.use_lock(%le, Release, %c1)
      aie.end
    }
  }
}

// -----

// Memtile (1, 1) S2MM 5 BD 15 at 0x021485D0; own memory (window 7)
// (0xB00000 + 7 * 0x300000 + 8192) / 4 = 0x800800; locks 448 and 449.
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<11xi32> = dense<[0, -33554176, 469764995, 0, 0, 0, 0, 0, 0, 0, 0]>
// CHECK-LABEL: @memtile_own
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 34899408 : ui32} : memref<11xi32>
// CHECK: aiex.npu.maskwrite32(%c34899408_i32, %c8390656_i32, %c16777215_i32)

aie.device(npu3) {
  %m11 = aie.tile(1, 1)
  %own = aie.buffer(%m11) {address = 8192 : i32} : memref<256xi32>
  %l0 = aie.lock(%m11, 0)
  %l1 = aie.lock(%m11, 1)
  aie.runtime_sequence @memtile_own() {
    %c1 = arith.constant 1 : i32
    %task = aiex.dma_configure_task(%m11, S2MM, 5) {
      aie.use_lock(%l0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%own : memref<256xi32> offset = 0 len = 256) {bd_id = 15 : i32}
      aie.use_lock(%l1, Release, %c1)
      aie.end
    }
  }
}

// -----

// Memtile (1, 1) MM2S 1 BD 0 at 0x02108F00; D0 pads in word 7 (0x302);
// len 84 = 4 * (2 + 16 + 3).
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<11xi32> = dense<[0, 84, 0, 32, 0, 0, 0, 770, 0, 2097664, 0]>
// CHECK-LABEL: @memtile_pad
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 34639616 : ui32} : memref<11xi32>
// CHECK: aiex.npu.maskwrite32(%c34639616_i32, %c8388608_i32, %c16777215_i32)

aie.device(npu3) {
  %m11 = aie.tile(1, 1)
  %own = aie.buffer(%m11) {address = 0 : i32} : memref<256xi32>
  aie.runtime_sequence @memtile_pad() {
    %task = aiex.dma_configure_task(%m11, MM2S, 1) {
      aie.dma_bd(%own : memref<256xi32> offset = 0 len = 84 sizes = [1, 1, 4, 16] strides = [0, 0, 32, 1] pad [<const_pad_before=0, const_pad_after=0>, <const_pad_before=2, const_pad_after=3>]) {bd_id = 0 : i32}
      aie.end
    }
  }
}

// -----

// Core tile (0, 2) BD 4 at 0x006AC080 (core row 2 is address row 6); address
// field 1024 / 4 in bits [30:16].
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<6xi32> = dense<[128, 0, 2097152, 2097152, 67108864, 294882]>
// CHECK-LABEL: @core
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 6996096 : ui32} : memref<6xi32>
// CHECK: aiex.npu.maskwrite32(%c6996096_i32, %c16777216_i32, %c2147418112_i32)

aie.device(npu3) {
  %t02 = aie.tile(0, 2)
  %buf = aie.buffer(%t02) {address = 1024 : i32} : memref<256xi32>
  %l2 = aie.lock(%t02, 2)
  %l3 = aie.lock(%t02, 3)
  aie.runtime_sequence @core() {
    %c1 = arith.constant 1 : i32
    %task = aiex.dma_configure_task(%t02, MM2S, 0) {
      aie.use_lock(%l2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 128 sizes = [1, 1, 8, 16] strides = [0, 0, 32, 1]) {bd_id = 4 : i32}
      aie.use_lock(%l3, Release, %c1)
      aie.end
    }
  }
}

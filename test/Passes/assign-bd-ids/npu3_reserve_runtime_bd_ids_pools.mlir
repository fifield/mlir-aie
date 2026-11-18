//===- npu3_reserve_runtime_bd_ids_pools.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-reserve-runtime-bd-ids --aie-assign-bd-ids \
// RUN:   --split-input-file %s | FileCheck %s

// AIE4: every memtile channel and direction has a private 16-BD pool, so an id
// a runtime task pins is reserved only in its own channel's pool.

// CHECK-LABEL: @memtile_other_pool
// CHECK:       aie.tile(0, 1) {aiex.reserved_channel_bd_ids = {MM2S_0 = array<i32: 0>}}
// CHECK-NOT:   aiex.reserved_bd_ids
// CHECK:       aie.dma_start(S2MM, 0
// CHECK:       aie.dma_bd({{.*}}) {bd_id = 0 : i32, next_bd_id = 1 : i32}
// CHECK-COUNT-14: aie.dma_bd(
// CHECK:       aie.dma_bd({{.*}}) {bd_id = 15 : i32, next_bd_id = 0 : i32}
module @memtile_other_pool {
  aie.device(npu3) {
    %mem = aie.tile(0, 1)
    %buf = aie.buffer(%mem) {address = 0 : i32} : memref<1024xi32>
    %m = aie.memtile_dma(%mem) {
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 16)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%buf : memref<1024xi32> offset = 16 len = 16)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%buf : memref<1024xi32> offset = 32 len = 16)
      aie.next_bd ^bd3
    ^bd3:
      aie.dma_bd(%buf : memref<1024xi32> offset = 48 len = 16)
      aie.next_bd ^bd4
    ^bd4:
      aie.dma_bd(%buf : memref<1024xi32> offset = 64 len = 16)
      aie.next_bd ^bd5
    ^bd5:
      aie.dma_bd(%buf : memref<1024xi32> offset = 80 len = 16)
      aie.next_bd ^bd6
    ^bd6:
      aie.dma_bd(%buf : memref<1024xi32> offset = 96 len = 16)
      aie.next_bd ^bd7
    ^bd7:
      aie.dma_bd(%buf : memref<1024xi32> offset = 112 len = 16)
      aie.next_bd ^bd8
    ^bd8:
      aie.dma_bd(%buf : memref<1024xi32> offset = 128 len = 16)
      aie.next_bd ^bd9
    ^bd9:
      aie.dma_bd(%buf : memref<1024xi32> offset = 144 len = 16)
      aie.next_bd ^bd10
    ^bd10:
      aie.dma_bd(%buf : memref<1024xi32> offset = 160 len = 16)
      aie.next_bd ^bd11
    ^bd11:
      aie.dma_bd(%buf : memref<1024xi32> offset = 176 len = 16)
      aie.next_bd ^bd12
    ^bd12:
      aie.dma_bd(%buf : memref<1024xi32> offset = 192 len = 16)
      aie.next_bd ^bd13
    ^bd13:
      aie.dma_bd(%buf : memref<1024xi32> offset = 208 len = 16)
      aie.next_bd ^bd14
    ^bd14:
      aie.dma_bd(%buf : memref<1024xi32> offset = 224 len = 16)
      aie.next_bd ^bd15
    ^bd15:
      aie.dma_bd(%buf : memref<1024xi32> offset = 240 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.runtime_sequence @seq() {
      %t = aiex.dma_configure_task(%mem, MM2S, 0) {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// CHECK-LABEL: @memtile_same_pool
// CHECK:       aie.tile(0, 1) {aiex.reserved_channel_bd_ids = {MM2S_0 = array<i32: 0, 3>, S2MM_1 = array<i32: 0>}}
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.dma_bd({{.*}}) {bd_id = 1 : i32, next_bd_id = 2 : i32}
// CHECK:       aie.dma_bd({{.*}}) {bd_id = 2 : i32, next_bd_id = 4 : i32}
// CHECK:       aie.dma_bd({{.*}}) {bd_id = 4 : i32, next_bd_id = 1 : i32}
// CHECK:       aie.dma_start(S2MM, 1
// CHECK:       aie.dma_bd({{.*}}) {bd_id = 1 : i32, next_bd_id = 1 : i32}
// CHECK:       aie.dma_start(S2MM, 0
// CHECK:       aie.dma_bd({{.*}}) {bd_id = 0 : i32, next_bd_id = 0 : i32}
module @memtile_same_pool {
  aie.device(npu3) {
    %mem = aie.tile(0, 1)
    %buf = aie.buffer(%mem) {address = 0 : i32} : memref<1024xi32>
    %m = aie.memtile_dma(%mem) {
      %0 = aie.dma_start(MM2S, 0, ^a, ^s1)
    ^a:
      aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 16)
      aie.next_bd ^b
    ^b:
      aie.dma_bd(%buf : memref<1024xi32> offset = 16 len = 16)
      aie.next_bd ^c
    ^c:
      aie.dma_bd(%buf : memref<1024xi32> offset = 32 len = 16)
      aie.next_bd ^a
    ^s1:
      %1 = aie.dma_start(S2MM, 1, ^d, ^s0)
    ^d:
      aie.dma_bd(%buf : memref<1024xi32> offset = 48 len = 16)
      aie.next_bd ^d
    ^s0:
      %2 = aie.dma_start(S2MM, 0, ^e, ^end)
    ^e:
      aie.dma_bd(%buf : memref<1024xi32> offset = 64 len = 16)
      aie.next_bd ^e
    ^end:
      aie.end
    }
    aie.runtime_sequence @seq() {
      %t = aiex.dma_configure_task(%mem, MM2S, 0) {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256) {bd_id = 0 : i32}
        aie.next_bd ^x
      ^x:
        aie.dma_bd(%buf : memref<1024xi32> offset = 256 len = 256) {bd_id = 3 : i32}
        aie.end
      }
      %u = aiex.dma_configure_task(%mem, S2MM, 1) {
        aie.dma_bd(%buf : memref<1024xi32> offset = 512 len = 256) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_start_task(%u)
      aiex.dma_await_task(%t)
      aiex.dma_await_task(%u)
    }
  }
}

// -----

// A shim BD id lies in one bank only, so the flat tile-wide form is used.

// CHECK-LABEL: @shim_bank
// CHECK:       aie.tile(0, 0) {aiex.reserved_bd_ids = array<i32: 16>}
// CHECK-NOT:   aiex.reserved_channel_bd_ids
// CHECK:       aie.dma_start(MM2S, 3
// CHECK:       aie.dma_bd({{.*}}) {bd_id = 17 : i32
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.dma_bd({{.*}}) {bd_id = 0 : i32
module @shim_bank {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    %ext = aie.external_buffer : memref<256xi32>
    aie.shim_dma(%shim) {
      %0 = aie.dma_start(MM2S, 3, ^a, ^m0)
    ^a:
      aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^end
    ^m0:
      %1 = aie.dma_start(MM2S, 0, ^b, ^end)
    ^b:
      aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.runtime_sequence @seq(%arg: memref<256xi32>) {
      %t = aiex.dma_configure_task(%shim, MM2S, 2) {
        aie.dma_bd(%arg : memref<256xi32> offset = 0 len = 256) {bd_id = 16 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

//===- npu3_bd_pool_ids.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Pinned BD ids must lie in the channel's pool. AIE4 shim: MM2S 0-1 and S2MM 0
// use BDs 0-15, MM2S 2-3 and S2MM 1 use BDs 16-31.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<64xi32>
  // expected-error@+1 {{bd_id 3 is not accessible from channel (MM2S, 2), whose BD pool spans ids 16 to 31}}
  %dma = aie.shim_dma(%shim) {
    %0 = aie.dma_start(MM2S, 2, ^bd, ^end)
  ^bd:
    // expected-note@+1 {{this BD}}
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64) {bd_id = 3 : i32}
    aie.next_bd ^bd
  ^end:
    aie.end
  }
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<64xi32>
  // expected-error@+1 {{next_bd_id 20 is not accessible from channel (S2MM, 0), whose BD pool spans ids 0 to 15}}
  %dma = aie.shim_dma(%shim) {
    %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
  ^bd:
    // expected-note@+1 {{this BD}}
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64) {bd_id = 2 : i32, next_bd_id = 20 : i32}
    aie.next_bd ^bd
  ^end:
    aie.end
  }
}

// -----

// Ids inside each channel's bank are legal.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<64xi32>
  %dma = aie.shim_dma(%shim) {
    %0 = aie.dma_start(S2MM, 1, ^bd0, ^next)
  ^bd0:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64) {bd_id = 16 : i32, next_bd_id = 16 : i32}
    aie.next_bd ^bd0
  ^next:
    %1 = aie.dma_start(MM2S, 1, ^bd1, ^end)
  ^bd1:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64) {bd_id = 15 : i32, next_bd_id = 15 : i32}
    aie.next_bd ^bd1
  ^end:
    aie.end
  }
}

// -----

// Each memtile channel and direction has a private 16-BD pool.
aie.device(npu3) {
  %mem = aie.tile(0, 1)
  %buf = aie.buffer(%mem) : memref<64xi32>
  %dma = aie.memtile_dma(%mem) {
    %0 = aie.dma_start(MM2S, 0, ^a, ^next)
  ^a:
    aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64) {bd_id = 15 : i32}
    aie.next_bd ^a
  ^next:
    %1 = aie.dma_start(S2MM, 0, ^b, ^end)
  ^b:
    aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64) {bd_id = 15 : i32}
    aie.next_bd ^b
  ^end:
    aie.end
  }
}

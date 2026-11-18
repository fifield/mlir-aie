//===- npu3_bd_pools_errors.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids --verify-diagnostics --split-input-file %s

// AIE4 BD pools: pinned ids must lie in the channel's pool, collide only within
// a pool, and a pool holds 16 BDs.

// Shim S2MM channel 1 uses the upper bank, BDs 16-31.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %ext = aie.external_buffer : memref<256xi32>
  // expected-error@+1 {{bd_id 3 is not accessible from channel (S2MM, 1), whose BD pool spans ids 16 to 31}}
  aie.shim_dma(%shim) {
    %0 = aie.dma_start(S2MM, 1, ^bd0, ^end)
  ^bd0:
    // expected-note@+1 {{this BD}}
    aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256) {bd_id = 3 : i32}
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

// Shim MM2S channel 1 and S2MM channel 0 share the lower bank.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %ext = aie.external_buffer : memref<256xi32>
  aie.shim_dma(%shim) {
    %0 = aie.dma_start(MM2S, 1, ^bd0, ^s0)
  ^bd0:
    aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256) {bd_id = 4 : i32}
    aie.next_bd ^end
  ^s0:
    %1 = aie.dma_start(S2MM, 0, ^bd1, ^end)
  ^bd1:
    // expected-error@+1 {{assigned bd_id 4 is already used by another BD on this tile}}
    aie.dma_bd(%ext : memref<256xi32> offset = 0 len = 256) {bd_id = 4 : i32}
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

aie.device(npu3) {
  %mem = aie.tile(0, 1)
  %a = aie.buffer(%mem) : memref<256xi32>
  aie.memtile_dma(%mem) {
    %0 = aie.dma_start(MM2S, 3, ^bd0, ^end)
  ^bd0:
    aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256) {bd_id = 2 : i32}
    aie.next_bd ^bd1
  ^bd1:
    // expected-error@+1 {{assigned bd_id 2 is already used by another BD on this tile}}
    aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256) {bd_id = 2 : i32}
    aie.next_bd ^bd0
  ^end:
    aie.end
  }
}

// -----

aie.device(npu3) {
  %mem = aie.tile(0, 1)
  %a = aie.buffer(%mem) : memref<256xi32>
  // expected-error@+1 {{BD pool for channel (S2MM, 2) exceeds limit of 16 BDs}}
  aie.memtile_dma(%mem) {
    %0 = aie.dma_start(S2MM, 2, ^bd0, ^end)
  ^bd0:
    aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd1
  ^bd1:
    aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd2
  ^bd2:
    aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd3
  ^bd3:
    aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd4
  ^bd4:
    aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd5
  ^bd5:
    aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd6
  ^bd6:
    aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd7
  ^bd7:
    aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd8
  ^bd8:
    aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd9
  ^bd9:
    aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd10
  ^bd10:
    aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd11
  ^bd11:
    aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd12
  ^bd12:
    aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd13
  ^bd13:
    aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd14
  ^bd14:
    aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd15
  ^bd15:
    aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd16
  ^bd16:
    // expected-note@+1 {{no space for this BD}}
    aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd0
  ^end:
    aie.end
  }
}

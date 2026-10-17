//===- aie4_bd_one_sided_locks.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file -convert-aie-to-transaction %s | FileCheck %s

// AIE4 BDs have an acquire-enable bit and release nothing when Lock_Rel_Value
// is 0, so a static BD may acquire without releasing or release without
// acquiring. A multi-BD transfer acquires on its
// first BD and releases on its last; a zero-length sync BD is lock-only.

// Core tile BD word 5: [24:18] Lock_Rel_Value, [16:13] Lock_Rel_ID,
// [12] Lock_Acq_Enable, [11:5] Lock_Acq_Value, [3:0] Lock_Acq_ID.
//   bd0 acquire-only, acq_ge(lock 3, 2):  0x7FC3 = acq id 3, value -2, enable 1,
//       release value 0 (release id = acquire id, unused)
//   bd1 release-only, rel(lock 5, 1):     0x4A005 = acquire enable 0, release id 5, value 1
//   bd2 zero-length, release-only:        same lock word, length 0
// CHECK-LABEL: aie.device(npu3)
// CHECK: dense<[16, 0, 65536, 1, 1, 32707]>
// CHECK: dense<[1048592, 0, 65536, 1, 1, 303109]>
// CHECK: dense<[2097152, 0, 65536, 1, 1, 303109]>
module {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) { address = 0 : i32, sym_name = "b" } : memref<256xi32>
    %la = aie.lock(%t, 3) { init = 1 : i32 }
    %lr = aie.lock(%t, 5) { init = 0 : i32 }
    %m = aie.mem(%t) {
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%la, AcquireGreaterEqual, %c2)
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 16) { bd_id = 0 : i32 }
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%b : memref<256xi32> offset = 16 len = 16) { bd_id = 1 : i32 }
      aie.use_lock(%lr, Release, %c1)
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%b : memref<256xi32> offset = 32 len = 0) { bd_id = 2 : i32 }
      aie.use_lock(%lr, Release, %c1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}

// -----

// Mem tile BD word 2: [29:20] Lock_Acq_ID, [17:11] Lock_Rel_Value,
// [10:1] Lock_Rel_ID, [0] Lock_Acq_Enable (word 1 [31:25] Lock_Acq_Value).
// Lock ids carry the base index of the lock's own tile: local (0,1) locks are
// 7 * 64 + id, east (1,1) locks 8 * 64 + id.
//   bd0 acq_ge(local lock 2), rel(east lock 3): 0x1C200C07 = acq id 450,
//       release id 515, value 1 (each lock offset by its own tile)
//   bd1 acquire-only zero-length acq_ge(local lock 2): 0x1C200385 = acq id 450,
//       release value 0
// CHECK-LABEL: aie.device(npu3)
// CHECK: dense<[8388608, -33554416, 471862279, 1, 1, 1, 1, 0, 0, 0, 0]>
// CHECK: dense<[8388608, -33554432, 471860101, 1, 1, 1, 1, 0, 0, 0, 0]>
module {
  aie.device(npu3) {
    %t = aie.tile(0, 1)
    %e = aie.tile(1, 1)
    %b = aie.buffer(%t) { address = 0 : i32, sym_name = "b" } : memref<256xi32>
    %la = aie.lock(%t, 2) { init = 1 : i32 }
    %lr = aie.lock(%e, 3) { init = 0 : i32 }
    %m = aie.memtile_dma(%t) {
      %c1 = arith.constant 1 : i32
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%la, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 16) { bd_id = 0 : i32 }
      aie.use_lock(%lr, Release, %c1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%la, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 0) { bd_id = 1 : i32 }
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}

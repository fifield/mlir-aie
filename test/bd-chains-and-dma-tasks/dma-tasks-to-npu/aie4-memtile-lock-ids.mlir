//===- aie4-memtile-lock-ids.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s --check-prefix=WRITEBD
// RUN: aie-opt --aie-dma-tasks-to-npu --aie-dma-to-npu %s | FileCheck %s --check-prefix=WORDS

// An AIE4 memtile's own locks start at local lock index 448
// (getLockLocalBaseIndex), so its locks 2 and 3 are hardware locks 450 and
// 451. aie-dma-tasks-to-npu applies that base once; the writebd lowering must
// not add it again.

// WRITEBD: aiex.npu.writebd {{.*}}lock_acq_enable = 1 : i32, lock_acq_id = 450 : i32, lock_acq_val = -1 : i32, lock_rel_id = 451 : i32, lock_rel_val = 1 : i32

// Word 2 = Lock_Acq_ID 450 [29:20] | Lock_Rel_Value 1 [17:11] |
// Lock_Rel_ID 451 [10:1] | Lock_Acq_Enable [0] = 471862151.
// WORDS: memref.global "private" constant @blockwrite_data_0 : memref<11xi32> = dense<[0, -33554424, 471862151, 0, 0, 0, 0, 0, 0, 0, 0]>

module {
  aie.device(npu3) {
    %tile = aie.tile(0, 1)
    %buf = aie.buffer(%tile) {address = 0 : i32} : memref<8xi32>
    %prod = aie.lock(%tile, 2) {init = 1 : i32}
    %cons = aie.lock(%tile, 3) {init = 0 : i32}
    aie.runtime_sequence() {
      %t = aiex.dma_configure_task(%tile, S2MM, 0) {
        %c1 = arith.constant 1 : i32
        aie.use_lock(%prod, AcquireGreaterEqual, %c1)
        aie.dma_bd(%buf : memref<8xi32> offset = 0 len = 8) {bd_id = 0 : i32}
        aie.use_lock(%cons, Release, %c1)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

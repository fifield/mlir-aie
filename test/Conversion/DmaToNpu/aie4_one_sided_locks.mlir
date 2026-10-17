//===- aie4_one_sided_locks.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// AIE4 runtime-sequence BDs may acquire without releasing or release without
// acquiring, and each lock id carries the base
// index of its own tile (local (0,1): 7 * 64 + id; east (1,1): 8 * 64 + id).

// CHECK: aiex.npu.writebd {bd_id = 0 : i32
// CHECK-SAME: lock_acq_enable = 1 : i32, lock_acq_id = 450 : i32, lock_acq_val = -4 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32
// CHECK: aiex.npu.writebd {bd_id = 1 : i32
// CHECK-SAME: lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 515 : i32, lock_rel_val = 1 : i32
// CHECK: aiex.npu.writebd {bd_id = 2 : i32
// CHECK-SAME: lock_acq_enable = 1 : i32, lock_acq_id = 450 : i32, lock_acq_val = -1 : i32, lock_rel_id = 515 : i32, lock_rel_val = 1 : i32
module {
  aie.device(npu3) {
    %t = aie.tile(0, 1)
    %e = aie.tile(1, 1)
    %b = aie.buffer(%t) { address = 0 : i32, sym_name = "b" } : memref<256xi32>
    %la = aie.lock(%t, 2) { init = 1 : i32 }
    %lr = aie.lock(%e, 3) { init = 0 : i32 }
    aie.runtime_sequence @seq(%arg0: memref<256xi32>) {
      %c1 = arith.constant 1 : i32
      %c4 = arith.constant 4 : i32
      %t0 = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.use_lock(%la, AcquireGreaterEqual, %c4)
        aie.dma_bd(%b : memref<256xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.end
      }
      %t1 = aiex.dma_configure_task(%t, S2MM, 1) {
        aie.dma_bd(%b : memref<256xi32> offset = 0 len = 16) {bd_id = 1 : i32}
        aie.use_lock(%lr, Release, %c1)
        aie.end
      }
      %t2 = aiex.dma_configure_task(%t, S2MM, 2) {
        aie.use_lock(%la, AcquireGreaterEqual, %c1)
        aie.dma_bd(%b : memref<256xi32> offset = 0 len = 16) {bd_id = 2 : i32}
        aie.use_lock(%lr, Release, %c1)
        aie.end
      }
      aiex.dma_start_task(%t0)
      aiex.dma_start_task(%t1)
      aiex.dma_start_task(%t2)
    }
  }
}

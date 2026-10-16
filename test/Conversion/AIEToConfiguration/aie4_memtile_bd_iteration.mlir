//===- aie4_memtile_bd_iteration.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file -convert-aie-to-transaction %s | FileCheck %s

// AIE4 mem tile BD iteration lowers into the 11-word BD: word 6 is the
// iteration step (in 32-bit words), word 7 bits [29:22] Iteration_Current,
// and word 10 Iteration_Wrap - 1. Both iteration fields are 8 bits wide.

// CHECK-LABEL: aie.device(npu3)
// CHECK: dense<[8388608, -33554416, 469764993, 1, 1, 1, 16, 109051904, 0, 0, 51]>
module {
  aie.device(npu3) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) { address = 0 : i32, sym_name = "b" } : memref<4096xi32>
    %lk = aie.lock(%t, 0) { init = 1 : i32 }
    %m = aie.memtile_dma(%t) {
      %c1 = arith.constant 1 : i32
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lk, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b : memref<4096xi32> offset = 0 len = 16) { bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 52, stride = 16, current = 26> }
      aie.use_lock(%lk, Release, %c1)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// Full 8-bit range: size 256 encodes as wrap 255, current 255.
// CHECK-LABEL: aie.device(npu3)
// CHECK: dense<[8388608, -33554416, 469764993, 1, 1, 1, 16, 1069547520, 0, 0, 255]>
module {
  aie.device(npu3) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) { address = 0 : i32, sym_name = "b" } : memref<4096xi32>
    %lk = aie.lock(%t, 0) { init = 1 : i32 }
    %m = aie.memtile_dma(%t) {
      %c1 = arith.constant 1 : i32
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lk, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b : memref<4096xi32> offset = 0 len = 16) { bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 256, stride = 16, current = 255> }
      aie.use_lock(%lk, Release, %c1)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

//===- aie4_dma_bd_d0_stride.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// AIE4 BDs have no D0 step (aie-rt silently drops one), so a non-unit
// innermost stride is an error.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu3_1col) {
    %m = aie.tile(0, 1)
    %buf = aie.buffer(%m) {address = 0 : i32} : memref<16xi32>
    %mt = aie.memtile_dma(%m) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{AIE4 D0 stride is fixed by hardware and must be 1.}}
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16 sizes = [4, 4] strides = [1, 4])
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}

// -----

// A core tile BD is the same.
module {
  aie.device(npu3_1col) {
    %c = aie.tile(0, 2)
    %buf = aie.buffer(%c) {address = 4096 : i32} : memref<16xi32>
    %mem = aie.mem(%c) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{AIE4 D0 stride is fixed by hardware and must be 1.}}
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16 sizes = [2, 8] strides = [1, 2])
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}

// -----

// A unit innermost stride, or an innermost dimension of one element, is fine.
module {
  aie.device(npu3_1col) {
    %m = aie.tile(0, 1)
    %buf = aie.buffer(%m) {address = 0 : i32} : memref<16xi32>
    %mt = aie.memtile_dma(%m) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16 sizes = [2, 2, 4] strides = [4, 8, 1])
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16 sizes = [16, 1] strides = [1, 4])
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}

// -----

// On other targets a strided D0 remains valid.
module {
  aie.device(npu2_1col) {
    %m = aie.tile(0, 1)
    %buf = aie.buffer(%m) {address = 0 : i32} : memref<16xi32>
    %mt = aie.memtile_dma(%m) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16 sizes = [4, 4] strides = [1, 4])
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}

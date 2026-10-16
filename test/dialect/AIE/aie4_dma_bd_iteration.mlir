//===- aie4_dma_bd_iteration.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// AIE4 BDs support iteration. Mem tile Iteration_Wrap/Iteration_Current are
// 8-bit fields (size up to 256); core and shim tile fields are 6-bit (size up
// to 64).

// Mem tile: full 8-bit range with a nonzero starting iteration.
module {
  aie.device(npu3) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) : memref<1024xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<1024xi32> offset = 0 len = 4) { iteration = #aie.bd_iteration<size = 256, stride = 4, current = 255> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// Mem tile: size above the 8-bit range.
module {
  aie.device(npu3) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) : memref<1024xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{BD iteration size must be in [1, 256]}}
      aie.dma_bd(%b : memref<1024xi32> offset = 0 len = 4) { iteration = #aie.bd_iteration<size = 257, stride = 4, current = 0> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// Core tile: full 6-bit range.
module {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.mem(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 4) { iteration = #aie.bd_iteration<size = 64, stride = 4, current = 63> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// Core tile: size above the 6-bit range.
module {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<256xi32>
    %m = aie.mem(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // expected-error @+1 {{BD iteration size must be in [1, 64]}}
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 4) { iteration = #aie.bd_iteration<size = 65, stride = 4, current = 0> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

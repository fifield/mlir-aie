//===- bad_shared_bd_pool_aie4.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A BD cannot be reached from channels with different BD pools. AIE4 shim
// MM2S 0-1 share BDs 0-15, MM2S 2-3 BDs 16-31.

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buf = aie.external_buffer : memref<256xi32>
  // expected-error@+1 {{'aie.shim_dma' op BD is reached from channels (MM2S, 0) and (MM2S, 2), which draw from different BD pools; one BD cannot serve both}}
  %dma = aie.shim_dma(%shim) {
    %0 = aie.dma_start(MM2S, 0, ^bd0, ^next)
  ^next:
    %1 = aie.dma_start(MM2S, 2, ^bd0, ^end)
  ^bd0:
    // expected-note@+1 {{this BD}}
    aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buf = aie.external_buffer : memref<256xi32>
  // expected-error@+1 {{BD is reached from channels (MM2S, 2) and (MM2S, 0)}}
  %dma = aie.shim_dma(%shim) {
    %0 = aie.dma_start(MM2S, 2, ^bd0, ^next)
  ^next:
    %1 = aie.dma_start(MM2S, 0, ^bd0, ^end)
  ^bd0:
    // expected-note@+1 {{this BD}}
    aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

// The second chain joins the first one BD in.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buf = aie.external_buffer : memref<256xi32>
  // expected-error@+1 {{BD is reached from channels (S2MM, 0) and (S2MM, 1)}}
  %dma = aie.shim_dma(%shim) {
    %0 = aie.dma_start(S2MM, 0, ^bd0, ^next)
  ^next:
    %1 = aie.dma_start(S2MM, 1, ^bd1, ^end)
  ^bd0:
    aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bd1
  ^bd1:
    // expected-note@+1 {{this BD}}
    aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

// Each AIE4 memtile channel and direction has a private pool.
aie.device(npu3) {
  %mt = aie.tile(0, 1)
  %buf = aie.buffer(%mt) {address = 0 : i32} : memref<256xi32>
  // expected-error@+1 {{BD is reached from channels (MM2S, 0) and (MM2S, 1)}}
  %dma = aie.memtile_dma(%mt) {
    %0 = aie.dma_start(MM2S, 0, ^bd0, ^next)
  ^next:
    %1 = aie.dma_start(MM2S, 1, ^bd0, ^end)
  ^bd0:
    // expected-note@+1 {{this BD}}
    aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

// Channels of one pool may share a BD.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buf = aie.external_buffer : memref<256xi32>
  %dma = aie.shim_dma(%shim) {
    %0 = aie.dma_start(MM2S, 0, ^bd0, ^next)
  ^next:
    %1 = aie.dma_start(MM2S, 1, ^bd0, ^end)
  ^bd0:
    aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

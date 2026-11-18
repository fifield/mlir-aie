//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-assign-runtime-sequence-bd-ids %s

// On AIE4 a pinned runtime BD id must lie in its channel's pool.
module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.runtime_sequence @shim_s2mm_1(%arg0 : memref<1024xi32>) {
      %t = aiex.dma_configure_task(%shim, S2MM, 1) {
        // expected-error@+1 {{cannot be submitted on channel 1 (S2MM) of tile (0,0), whose BD pool holds ids 16 to 31}}
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256) {bd_id = 3 : i32}
        aie.end
      }
    }
  }
}

// -----

module {
  aie.device(npu3) {
    %mem = aie.tile(0, 1)
    %buf = aie.buffer(%mem) : memref<1024xi32>
    aie.runtime_sequence @memtile_local() {
      %t = aiex.dma_configure_task(%mem, MM2S, 4) {
        // expected-error@+1 {{cannot be submitted on channel 4 (MM2S) of tile (0,1), whose BD pool holds ids 0 to 15}}
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256) {bd_id = 16 : i32}
        aie.end
      }
    }
  }
}

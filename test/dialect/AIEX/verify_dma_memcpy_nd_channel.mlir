// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --verify-diagnostics %s
// RUN: aie-opt --aie-dma-to-npu --split-input-file --verify-diagnostics %s

// An unsupported allocation must be rejected before either lowering path.
aie.device(npu2) {
  %shim = aie.tile(0, 0)
  // expected-error@+1 {{unsupported DMA channel (ControlMM2S, 0) on tile (0, 0)}}
  aie.shim_dma_allocation @alloc (%shim, ControlMM2S, 0)
  aie.runtime_sequence(%arg: memref<64xi32>) {
    aiex.npu.dma_memcpy_nd (%arg[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @alloc, id = 0 : i64 } : memref<64xi32>
  }
}

// -----

// Check the BD pool before static lowering computes its address.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc (%shim, ControlMM2S, 0)
  aie.runtime_sequence(%arg: memref<64xi32>) {
    // expected-error@+1 {{BD ID 8 is not accessible from channel (ControlMM2S, 0)}}
    aiex.npu.dma_memcpy_nd (%arg[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @alloc, id = 8 : i64 } : memref<64xi32>
  }
}

// -----

// Check the BD pool before dynamic lowering computes its address.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc (%shim, TraceS2MM, 0)
  aie.runtime_sequence(%arg: memref<64xi32>, %n: i64) {
    // expected-error@+1 {{BD ID 4 is not accessible from channel (TraceS2MM, 0)}}
    aiex.npu.dma_memcpy_nd (%arg[0, 0, 0, 0][1, 1, 1, %n][0, 0, 0, 1]) { metadata = @alloc, id = 4 : i64 } : memref<64xi32>
  }
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc (%shim, MM2S, 2)
  aie.runtime_sequence(%arg: memref<64xi32>) {
    // expected-error@+1 {{BD ID 0 is not accessible from channel (MM2S, 2)}}
    aiex.npu.dma_memcpy_nd (%arg[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @alloc, id = 0 : i64 } : memref<64xi32>
  }
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc (%shim, MM2S, 2)
  aie.runtime_sequence(%arg: memref<64xi32>, %n: i64) {
    // expected-error@+1 {{BD ID 0 is not accessible from channel (MM2S, 2)}}
    aiex.npu.dma_memcpy_nd (%arg[0, 0, 0, 0][1, 1, 1, %n][0, 0, 0, 1]) { metadata = @alloc, id = 0 : i64 } : memref<64xi32>
  }
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc (%shim, MM2S, 0)
  aie.runtime_sequence(%arg: memref<64xi32>) {
    // expected-error@+1 {{BD ID 4294967296 is not accessible from channel (MM2S, 0)}}
    aiex.npu.dma_memcpy_nd (%arg[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @alloc, id = 4294967296 : i64 } : memref<64xi32>
  }
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc (%shim, MM2S, 0)
  aie.runtime_sequence(%arg: memref<64xi32>) {
    // expected-error@+1 {{BD ID 18446744073709551615 is not accessible from channel (MM2S, 0)}}
    aiex.npu.dma_memcpy_nd (%arg[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @alloc, id = -1 : i64 } : memref<64xi32>
  }
}

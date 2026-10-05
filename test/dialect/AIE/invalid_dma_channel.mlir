// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// Validate the channel before HasValidBDs queries its pool.
aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<64xi32>
  aie.shim_dma(%shim) {
    // expected-error@+1 {{unsupported DMA channel (ControlMM2S, 0) on tile (0, 0)}}
    %start = aie.dma_start(ControlMM2S, 0, ^bd, ^end)
  ^bd:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<64xi32>
  aie.shim_dma(%shim) {
    // expected-error@+1 {{unsupported DMA channel (TraceS2MM, 0) on tile (0, 0)}}
    %start = aie.dma_start(TraceS2MM, 0, ^bd, ^end)
  ^bd:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<64xi32>
  aie.shim_dma(%shim) {
    // expected-error@+1 {{unsupported DMA channel (MM2S, 4) on tile (0, 0)}}
    %start = aie.dma_start(MM2S, 4, ^bd, ^end)
  ^bd:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<64xi32>
  aie.shim_dma(%shim) {
    // expected-error@+1 {{unsupported DMA channel (S2MM, -1) on tile (0, 0)}}
    %start = aie.dma_start(S2MM, -1, ^bd, ^end)
  ^bd:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<64xi32>
  aie.shim_dma(%shim) {
    // expected-error@+1 {{unsupported DMA channel (ControlMM2S, -1) on tile (0, 0)}}
    %start = aie.dma_start(ControlMM2S, -1, ^bd, ^end)
  ^bd:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

// Region-form data DMA must reject an out-of-range channel as well.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %buffer = aie.external_buffer : memref<64xi32>
  %lock = aie.lock(%shim, 0)
  aie.shim_dma(%shim) {
    // expected-error@+1 {{unsupported DMA channel (MM2S, 4) on tile (0, 0)}}
    %bad = aie.dma(MM2S, 4) [{
      %zero = arith.constant 0 : i32
      aie.use_lock(%lock, Acquire, %zero)
      aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
      aie.use_lock(%lock, Release, %zero)
    }]
    aie.end
  }
}

// -----

// Standalone allocations carry the same hardware channel constraints.
aie.device(npu2) {
  %shim = aie.tile(0, 0)
  // expected-error@+1 {{unsupported DMA channel (ControlMM2S, 0) on tile (0, 0)}}
  aie.shim_dma_allocation @alloc (%shim, ControlMM2S, 0)
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  // expected-error@+1 {{unsupported DMA channel (TraceS2MM, 1) on tile (0, 0)}}
  aie.shim_dma_allocation @alloc (%shim, TraceS2MM, 1)
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  // expected-error@+1 {{unsupported DMA channel (S2MM, 2) on tile (0, 0)}}
  aie.shim_dma_allocation @alloc (%shim, S2MM, 2)
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  // expected-error@+1 {{unsupported DMA channel (MM2S, -1) on tile (0, 0)}}
  aie.shim_dma_allocation @alloc (%shim, MM2S, -1)
}

// -----

// Valid specialized engines and the upper data bank remain accepted.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @control (%shim, ControlMM2S, 1)
  aie.shim_dma_allocation @trace (%shim, TraceS2MM, 0)
  aie.shim_dma_allocation @data (%shim, MM2S, 3)
}

// -----

// AIE4 core tiles have two S2MM channels but only one MM2S channel.
// TileLike's aggregate connection counts must preserve that asymmetry.
aie.device(npu3) {
  %core = aie.tile(0, 2)
  %buffer = aie.buffer(%core) : memref<64xi32>
  aie.mem(%core) {
    %s0 = aie.dma_start(S2MM, 0, ^bd0, ^next0)
  ^bd0:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^bd0
  ^next0:
    %s1 = aie.dma_start(S2MM, 1, ^bd1, ^next1)
  ^bd1:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^bd1
  ^next1:
    %m0 = aie.dma_start(MM2S, 0, ^bd2, ^end)
  ^bd2:
    aie.dma_bd(%buffer : memref<64xi32> offset = 0 len = 64)
    aie.next_bd ^bd2
  ^end:
    aie.end
  }
}

// -----

// Allocation indices are i64; reject values that would wrap to a valid i32
// channel before adapting to the target model's channel representation.
aie.device(npu2) {
  %shim = aie.tile(0, 0)
  // expected-error@+1 {{unsupported DMA channel (MM2S, 4294967296) on tile (0, 0)}}
  aie.shim_dma_allocation @alloc (%shim, MM2S, 4294967296)
}

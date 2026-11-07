// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-dma-tasks-to-npu %s

// A four-entry pattern on an AIE4 shim is [iteration, d2, d1, d0]. When it was
// written as four addressing dims, the diagnostics point at the five-entry form.

// The outermost entry exceeds the iteration range.
module {
  aie.device(npu3) {
    %t = aie.tile(0, 0)
    aie.runtime_sequence(%out: memref<11059200xi32>) {
      %0 = aiex.dma_configure_task(%t, S2MM, 0) {
        // expected-error@+2 {{iteration count 720 exceeds the [1:64] range}}
        // expected-note@+1 {{write five entries [iteration, d3, d2, d1, d0] (here sizes = [1, 720, 3, 2, 1280] strides = [0, 5120, 3686400, 2560, 1]) or set the `iteration` attribute}}
        aie.dma_bd(%out : memref<11059200xi32> offset = 0 len = 5529600 sizes = [720, 3, 2, 1280] strides = [5120, 3686400, 2560, 1]) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}

// -----

// len covers all four entries.
module {
  aie.device(npu3) {
    %t = aie.tile(0, 0)
    aie.runtime_sequence(%out: memref<11059200xi32>) {
      %0 = aiex.dma_configure_task(%t, S2MM, 0) {
        // expected-error@+2 {{Buffer descriptor length does not match}}
        // expected-note@+1 {{(here sizes = [1, 4, 3, 2, 1280] strides = [0, 5120, 3686400, 2560, 1])}}
        aie.dma_bd(%out : memref<11059200xi32> offset = 0 len = 30720 sizes = [4, 3, 2, 1280] strides = [5120, 3686400, 2560, 1]) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}

// -----

// len leaves the outermost entry out: it lowers as an iteration, which a
// single push runs once.
module {
  aie.device(npu3) {
    %t = aie.tile(0, 0)
    aie.runtime_sequence(%out: memref<11059200xi32>) {
      %0 = aiex.dma_configure_task(%t, S2MM, 0) {
        // expected-warning@+2 {{iteration dimension of size 4 is pushed with repeat_count 0}}
        // expected-note@+1 {{(here sizes = [1, 4, 3, 2, 1280] strides = [0, 5120, 3686400, 2560, 1])}}
        aie.dma_bd(%out : memref<11059200xi32> offset = 0 len = 7680 sizes = [4, 3, 2, 1280] strides = [5120, 3686400, 2560, 1]) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}

// -----

// A stride-0 outermost entry is a repeat, not a D3 candidate: no note.
module {
  aie.device(npu3) {
    %t = aie.tile(0, 0)
    aie.runtime_sequence(%out: memref<11059200xi32>) {
      %0 = aiex.dma_configure_task(%t, S2MM, 0) {
        // expected-warning@+1 {{iteration dimension of size 4 is pushed with repeat_count 0}}
        aie.dma_bd(%out : memref<11059200xi32> offset = 0 len = 7680 sizes = [4, 3, 2, 1280] strides = [0, 3686400, 2560, 1]) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu --split-input-file %s
module {
  aie.device(npu1) {
    %tile = aie.tile(0, 1)
    %buf = aie.buffer(%tile) {address = 0 : i32} : memref<48xi32>
    aie.runtime_sequence() {
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        // expected-error@+1 {{Padding is only supported for D0, D1 and D2.}}
        aie.dma_bd(%buf : memref<48xi32> offset = 0 len = 48 sizes = [1, 3, 2, 2, 4] strides = [0, 16, 8, 4, 1] pad [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 1, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}
// -----
module {
  aie.device(npu1) {
    %tile = aie.tile(0, 1)
    %buf = aie.buffer(%tile) {address = 0 : i32} : memref<48xi32>
    aie.runtime_sequence() {
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        // expected-error@+1 {{Padding is only supported for D0, D1 and D2.}}
        aie.dma_bd(%buf : memref<48xi32> offset = 0 len = 48 sizes = [1, 3, 2, 2, 4] strides = [0, 16, 8, 4, 1] pad [<const_pad_before = 0, const_pad_after = 1>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

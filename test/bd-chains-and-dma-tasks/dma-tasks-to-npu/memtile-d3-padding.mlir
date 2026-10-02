// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s
//
// Zero padding placeholders for D3 and iteration must not index beyond the
// hardware padding fields, including when only the innermost pad is supplied.
// CHECK-COUNT-2: aiex.npu.writebd
module {
  aie.device(npu1) {
    %tile = aie.tile(0, 1)
    %buf = aie.buffer(%tile) {address = 0 : i32} : memref<48xi32>
    aie.runtime_sequence() {
      %full = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<48xi32> offset = 0 len = 48 sizes = [1, 3, 2, 2, 4] strides = [0, 16, 8, 4, 1] pad [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>]) {bd_id = 0 : i32}
        aie.end
      }
      %inner = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<48xi32> offset = 0 len = 48 sizes = [1, 3, 2, 2, 4] strides = [0, 16, 8, 4, 1] pad [<const_pad_before = 0, const_pad_after = 0>]) {bd_id = 1 : i32}
        aie.end
      }
    }
  }
}

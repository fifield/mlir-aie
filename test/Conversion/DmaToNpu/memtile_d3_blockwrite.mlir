//===- memtile_d3_blockwrite.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-buffer-addresses --aie-dma-tasks-to-npu --aie-dma-to-npu %s | FileCheck %s

// End-to-end (dma_configure_task -> writebd -> blockwrite) check that an AIE2
// (npu1) MemTile 5-dim D3 access pattern survives all the way into the final BD
// register words. D2_Wrap (the D2 size, needed so the DMA advances D3) and
// D3_Stepsize are both real addressing fields here; earlier the AIE2 MemTile
// block-write encoder dropped them.
//
// Access pattern (outermost..innermost): iter=5 d3=3 d2=2 d1=2 d0=4, AIE2
// "actual-1" step encoding. The lowered DMA_BD0_* words for an AIE2 MemTile:
//   word2 D0_Wrap[26:17]=4                       -> 4<<17 = 524288
//   word3 D1_Wrap[26:17]=2, D1_Stepsize=3        -> (2<<17)|3 = 262147
//   word4 D2_Wrap[26:17]=2, D2_Stepsize=7        -> (2<<17)|7 = 262151
//   word5 D3_Stepsize[16:0]=15                    -> 15
//   word6 Iteration_Size[22:17]=4, Iter_Step=47  -> (4<<17)|47 = 524335
//   word7 Valid_BD[31]=1                          -> 0x80000000 = -2147483648

// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[48, 0, 524288, 262147, 262151, 15, 524335, -2147483648]>

module {
  aie.device(npu1) {
    %tile_0_1 = aie.tile(0, 1)
    %buf0 = aie.buffer(%tile_0_1) : memref<240xi32>

    aie.runtime_sequence() {
      %t0 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
          aie.dma_bd(%buf0 : memref<240xi32> offset = 0 len = 48 sizes = [5, 3, 2, 2, 4] strides = [48, 16, 8, 4, 1]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

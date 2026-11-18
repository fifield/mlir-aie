//===- dma_to_npu_dynamic_no_layout.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu --verify-diagnostics %s

// A runtime-valued dma_memcpy_nd packs its BD words through the target's BD
// register layout. AIE1 models none, so the lowering rejects the tile, as the
// static path does, rather than dereferencing the missing layout.

module {
  aie.device(xcvc1902) {
    %t = aie.tile(2, 0)
    aie.shim_dma_allocation @a(%t, MM2S, 0)
    aie.runtime_sequence @s(%arg0: memref<64xi32>, %n: i64) {
      // expected-error@+2 {{'aiex.npu.dma_memcpy_nd' op has no buffer descriptor layout on this tile}}
      // expected-error@+1 {{failed to legalize operation 'aiex.npu.dma_memcpy_nd'}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 4, %n][0, 0, 4, 1]) {id = 0 : i64, metadata = @a} : memref<64xi32>
    }
  }
}

//===- npu3_bad_shim_dma_allocation_channel.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// An AIE4 shim has MM2S channels 0-3 and S2MM channels 0-1.

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  // expected-error@+1 {{'aie.shim_dma_allocation' op S2MM channel 5 is out of range: the tile has 2 S2MM channels}}
  aie.shim_dma_allocation @out(%shim, S2MM, 5)
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  // expected-error@+1 {{'aie.shim_dma_allocation' op MM2S channel 4 is out of range: the tile has 4 MM2S channels}}
  aie.shim_dma_allocation @in(%shim, MM2S, 4)
}

// -----

// The last channel of each direction is accepted.

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @in(%shim, MM2S, 3)
  aie.shim_dma_allocation @out(%shim, S2MM, 1)
}

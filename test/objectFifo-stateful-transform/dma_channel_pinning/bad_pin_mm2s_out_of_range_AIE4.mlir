//===- bad_pin_mm2s_out_of_range_AIE4.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// An npu3 (AIE4) core tile has 2 S2MM but only 1 MM2S DMA channel, so MM2S
// channel 1 does not exist and must be rejected.
//
// This is the direction that fails silently when the per-direction budgets
// are swapped: sizing MM2S with the S2MM count (2) lets channel 1 through,
// and the pass then emits aie.dma_start(MM2S, 1) on a tile that has no such
// channel -- an invalid design with no diagnostic. bad_channel_out_of_range
// cannot catch it because 99 is out of range under either budget, and no
// AIE2 device can catch it because both directions are 2 there.
//
// The positive counterpart (S2MM channel 1, which is legal) lives in
// pin_asymmetric_channels_AIE4.mlir.

// CHECK: error: 'aie.objectfifo' op pinned MM2S DMA channel 1 is out of range or already in use on this tile

module @bad_pin_mm2s_out_of_range_AIE4 {
  aie.device(npu3) {
    %t02 = aie.tile(0, 2)
    %t12 = aie.tile(1, 2)

    aie.objectfifo @of (%t02, {%t12}, 2 : i32) {prod_dma_channel = 1 : i32, cons_dma_channels = array<i32: 0>} : !aie.objectfifo<memref<16xi32>>
  }
}

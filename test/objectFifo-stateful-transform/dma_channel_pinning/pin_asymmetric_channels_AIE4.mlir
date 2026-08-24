//===- pin_asymmetric_channels_AIE4.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform %s | FileCheck %s

// AIE4 is the first target whose core tiles have a different number of DMA
// channels in each direction: 2 S2MM but only 1 MM2S. That makes it the only
// device that can catch a swapped per-direction channel budget -- on
// AIE1/AIE2/AIE2P both directions are 2, so an inverted query returns the
// right answer by accident.
//
// Pinning channel 1 is the discriminating case: legal for S2MM (channels 0
// and 1 exist) and illegal for MM2S (only channel 0 exists). See the
// companion negative case in bad_pin_mm2s_out_of_range_AIE4.mlir.

// S2MM channel 1 is in range on an npu3 core tile, so this must lower.
// CHECK-LABEL:   @pin_s2mm_channel1
// CHECK:           aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_1_2, DMA : 1)
// CHECK:             aie.dma_start(S2MM, 1,
module @pin_s2mm_channel1 {
  aie.device(npu3) {
    %t02 = aie.tile(0, 2)
    %t12 = aie.tile(1, 2)

    aie.objectfifo @of (%t02, {%t12}, 2 : i32) {prod_dma_channel = 0 : i32, cons_dma_channels = array<i32: 1>} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// MM2S channel 0 is the only legal producer channel on an npu3 core tile.
// CHECK-LABEL:   @pin_mm2s_channel0
// CHECK:             aie.dma_start(MM2S, 0,
module @pin_mm2s_channel0 {
  aie.device(npu3) {
    %t02 = aie.tile(0, 2)
    %t12 = aie.tile(1, 2)

    aie.objectfifo @of (%t02, {%t12}, 2 : i32) {prod_dma_channel = 0 : i32, cons_dma_channels = array<i32: 0>} : !aie.objectfifo<memref<16xi32>>
  }
}

//===- reset_npu3.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-core-reset --aie-lower-dma-channel-reset %s | FileCheck %s
// RUN: aie-opt --aie-lower-core-reset --aie-lower-dma-channel-reset %s | aie-translate --aie-npu-to-binary -aie-output-binary=false | FileCheck %s --check-prefix=TXN

// AIE4 reset is bit 1 of Core_Control (0xA3000) and of the DMA _Ctrl registers
// (core DMA_S2MM_0_Ctrl 0xACE00, memtile DMA_MM2S_1_Ctrl 0x9E68). Core tile
// (0, 2) is address row 6, memtile (0, 1) is row 1.

// TXN: 006A3000
// TXN: 006A3000
// TXN: 006ACE00
// TXN: 006ACE00
// TXN: 00109E68
// TXN: 00109E68

// CHECK-LABEL: aie.device(npu3)
// CHECK: %[[A0:.*]] = arith.constant 667648 : i32
// CHECK: %[[V0:.*]] = arith.constant 2 : i32
// CHECK: %[[M0:.*]] = arith.constant 2 : i32
// CHECK: aiex.npu.maskwrite32(%[[A0]], %[[V0]], %[[M0]]) {column = 0 : i32, row = 2 : i32}
// CHECK: %[[A1:.*]] = arith.constant 667648 : i32
// CHECK: %[[V1:.*]] = arith.constant 0 : i32
// CHECK: %[[M1:.*]] = arith.constant 2 : i32
// CHECK: aiex.npu.maskwrite32(%[[A1]], %[[V1]], %[[M1]]) {column = 0 : i32, row = 2 : i32}
// CHECK: %[[A2:.*]] = arith.constant 708096 : i32
// CHECK: %[[V2:.*]] = arith.constant 2 : i32
// CHECK: %[[M2:.*]] = arith.constant 2 : i32
// CHECK: aiex.npu.maskwrite32(%[[A2]], %[[V2]], %[[M2]]) {column = 0 : i32, row = 2 : i32}
// CHECK: arith.constant 708096 : i32
// CHECK: aiex.npu.maskwrite32
// CHECK: %[[A3:.*]] = arith.constant 40552 : i32
// CHECK: %[[V3:.*]] = arith.constant 2 : i32
// CHECK: %[[M3:.*]] = arith.constant 2 : i32
// CHECK: aiex.npu.maskwrite32(%[[A3]], %[[V3]], %[[M3]]) {column = 0 : i32, row = 1 : i32}
// CHECK: arith.constant 40552 : i32
// CHECK: aiex.npu.maskwrite32

module {
  aie.device(npu3) {
    %core_tile = aie.tile(0, 2)
    %mem_tile = aie.tile(0, 1)
    aie.runtime_sequence() {
      aiex.core_reset(%core_tile)
      aiex.dma_channel_reset(%core_tile, S2MM, 0)
      aiex.dma_channel_reset(%mem_tile, MM2S, 1)
    }
  }
}

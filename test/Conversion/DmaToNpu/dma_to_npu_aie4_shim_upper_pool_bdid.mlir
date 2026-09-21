//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// Load-bearing: pins how an AIE4b (npu3) shim BD id from the *upper* pool is
// encoded, in both places it appears.
//
// An AIE4b shim has two shared BD pools (AIE4TargetModel::getBdPoolId):
// lower = ids 0..15 on MM2S ch0/1 and S2MM ch0, upper = ids 16..31 on MM2S
// ch>=2 and S2MM ch1 (getBdPoolStartId returns 16 for pool 1). The bd_id the
// allocator hands out is global across both pools, but the hardware wants a
// *bank-local* index in both consumers:
//
//   - the BD register block address, where AIE4TargetModel::getDmaBdAddress
//     subtracts the pool start explicitly, and
//   - the DMA task queue command word, whose Start_BD_ID field is 4 bits wide
//     on an AIE4 shim (register database aie_registers_aie4.json), so the
//     `bd_id & 0xF` in PushQueuetoWrite32Pattern performs the same
//     subtraction: 18 & 0xF == 18 - 16 == 2.
//
// That second one reads like a truncation bug and is not: widening the mask
// (e.g. to the 6-bit Start BD ID of the architecture specification's generic
// task format) would write the global id 18 into a field that means index 18
// of a 16-entry bank. This test fails if either consumer stops agreeing on
// the bank-local index.

// RUN: aie-opt --split-input-file --aie-dma-tasks-to-npu --aie-dma-to-npu %s | FileCheck %s

// MM2S channel 2 draws from the upper pool, so bd_id 18 is bank-local 2.
// The BD register block sits in the upper bank (0x58060), and the queue
// command word carries 2, not 18.

// CHECK-LABEL: @upper_pool_bd18
// CHECK: aiex.npu.blockwrite
// CHECK-SAME: address = 360544
// CHECK:      %[[CMD:.*]] = arith.constant 2 : i32
// CHECK-NEXT: %[[QADDR:.*]] = arith.constant 361812 : i32
// CHECK-NEXT: aiex.npu.write32(%[[QADDR]], %[[CMD]])
module @upper_pool_bd18 {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task(%shim, MM2S, 2) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64) {bd_id = 18 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// Contrast: a lower-pool id needs no conversion, and lands in a different BD
// register bank. Only the bank address is pinned here -- the queue word for a
// lower-pool BD is just the id itself, so it cannot distinguish "converted"
// from "not converted" the way the upper-pool case above does.

// CHECK-LABEL: @lower_pool_bd5
// CHECK: aiex.npu.blockwrite
// CHECK-SAME: address = 98544
module @lower_pool_bd5 {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64) {bd_id = 5 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

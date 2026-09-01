//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The dma_task dynamic BD-word encoder (rewriteSingleBDDynamic) must address
// the BD register bank of the descriptor's own channel and direction.
// ControlMM2S channel 1 owns a private eight-BD pool based at 0x58300
// (361216); the ordinary shim MM2S bank at 0x18000 (98304) is a different set
// of registers, and writing the BD words there would leave the address patch
// (which does resolve the private bank) patching a register nobody configured.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-substitute-shim-dma-allocations --aie-dma-tasks-to-npu %s | FileCheck %s

// CHECK-LABEL: @ctrl
// CHECK-NOT: aiex.npu.blockwrite_values(%c98304
// CHECK: aiex.npu.blockwrite_values(%c361216{{.*}} : i32) values
// CHECK: aiex.npu.address_patch({{.*}}) {addr = 361220 : ui32

module {
  aie.device(npu3) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_ctrl (%tile_0_0, ControlMM2S, 1)
    aie.runtime_sequence @ctrl(%in: memref<8192xi32>, %n: i64) {
      %t = aiex.dma_configure_task_for @of_ctrl {
        aie.dma_bd(%in : memref<8192xi32> offset = 0 len = 256 sizes = [1, 1, %n, 32] strides = [0, 0, 32, 1]) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

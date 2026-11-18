//===- offset_parameter_npu3.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics \
// RUN:   --pass-pipeline='builtin.module(aie-lower-scratchpad-parameters,aie.device(aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu))' %s \
// RUN:   | FileCheck %s

// AIE4 memtile BD address is in words at bit 0 of word 0: an i32 offset adds 1
// per element. MM2S 1 BD 2 is DMA_MM2S_1_BD2_0 (0x108F60).
// CHECK-LABEL: aie.device(npu3) @mem
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = 1085280 : ui32, func_arg = 1 : ui32
// CHECK-NOT: aiex.npu.maskwrite32
// CHECK: aiex.npu.push_queue
// expected-remark@+1 {{takes offsets in multiples of 4 elements}}
aiex.scratchpad_parameter @off : i32
aie.device(npu3) @mem {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) {address = 1024 : i32, sym_name = "b"} : memref<128xi32>
  aie.runtime_sequence(%a : memref<16xi32>) {
    %ta = aiex.dma_configure_task(%t, MM2S, 1) {
      aie.dma_bd(%buf : memref<128xi32> offset = 0 len = 16) {offset_parameter = @off, bd_id = 2 : i32}
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%ta)
    aiex.dma_await_task(%ta)
  }
}

// -----

// AIE4 core BD word address is at bit 16 of word 0, above the length: an i32
// offset adds 1 << 16; a maskwrite restores the length's low two bits.
// CHECK-LABEL: aie.device(npu3) @core_i32
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = [[W0:[0-9]+]] : ui32, func_arg = 65536 : ui32
// CHECK-NEXT: %[[A:.+]] = arith.constant [[W0]] : i32
// CHECK-NEXT: %[[V:.+]] = arith.constant 0 : i32
// CHECK-NEXT: %[[M:.+]] = arith.constant 3 : i32
// CHECK-NEXT: aiex.npu.maskwrite32(%[[A]], %[[V]], %[[M]])
aiex.scratchpad_parameter @off : i32
aie.device(npu3) @core_i32 {
  %t = aie.tile(0, 2)
  %buf = aie.buffer(%t) {address = 1024 : i32, sym_name = "b"} : memref<128xi32>
  aie.runtime_sequence(%a : memref<16xi32>) {
    %ta = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%buf : memref<128xi32> offset = 0 len = 16) {offset_parameter = @off, bd_id = 0 : i32}
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%ta)
    aiex.dma_await_task(%ta)
  }
}

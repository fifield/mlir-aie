//===- dma_to_npu_aie4_dynamic_private_bd_banks.mlir -----------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The dynamic (runtime SSA size/stride) shim-NOC dma_memcpy_nd lowering must
// address the BD register bank that belongs to the descriptor's own channel and
// direction. ControlMM2S and TraceS2MM own private BD pools, so a lowering that
// drops channel/direction writes the BD words into the ordinary MM2S bank at
// 0x18000 (98304) while the address patch still targets the private bank.

// RUN: aie-opt --split-input-file --aie-dma-to-npu %s | FileCheck %s

// ControlMM2S channel 1: the private pool base is 0x58300 (361216), and the
// address patch of word 1 lands at 0x58304 (361220) in the same bank.
// ControlMM2S is a transmit direction, so the queue push must NOT force
// issue_token (bit 31): the command word here is plain 0.
// CHECK-LABEL: @ctrl_mm2s_ch1
// CHECK-NOT: aiex.npu.blockwrite_values(%c98304
// CHECK: aiex.npu.blockwrite_values(%c361216{{.*}} : i32) values
// CHECK: aiex.npu.address_patch({{.*}}) {addr = 361220 : ui32
// CHECK-NOT: arith.constant -2147483648
module {
  aie.device(npu3) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0(%t, ControlMM2S, 1)
    aie.runtime_sequence @ctrl_mm2s_ch1(%arg0: memref<4096xi32>, %n: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, %n, 32][0, 0, 32, 1]) {id = 0 : i64, metadata = @alloc0} : memref<4096xi32>
    }
  }
}

// -----

// ControlMM2S channel 0: private pool base 0x18300 (99072), patch at 0x18304.
// CHECK-LABEL: @ctrl_mm2s_ch0
// CHECK-NOT: aiex.npu.blockwrite_values(%c98304
// CHECK: aiex.npu.blockwrite_values(%c99072{{.*}} : i32) values
// CHECK: aiex.npu.address_patch({{.*}}) {addr = 99076 : ui32
module {
  aie.device(npu3) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0(%t, ControlMM2S, 0)
    aie.runtime_sequence @ctrl_mm2s_ch0(%arg0: memref<4096xi32>, %n: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, %n, 32][0, 0, 32, 1]) {id = 0 : i64, metadata = @alloc0} : memref<4096xi32>
    }
  }
}

// -----

// TraceS2MM channel 0: private pool base 0x18480 (99456), patch at 0x18484.
// This one really is a receive channel, so issue_token stays forced on.
// CHECK-LABEL: @trace_s2mm_ch0
// CHECK-NOT: aiex.npu.blockwrite_values(%c98304
// CHECK: aiex.npu.blockwrite_values(%c99456{{.*}} : i32) values
// CHECK: aiex.npu.address_patch({{.*}}) {addr = 99460 : ui32
// CHECK-DAG: %[[Q:.*]] = arith.constant 99660 : i32
// CHECK-DAG: %[[CMD:.*]] = arith.constant -2147483648 : i32
// CHECK: aiex.npu.write32(%[[Q]], %[[CMD]]) : i32, i32
module {
  aie.device(npu3) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @alloc0(%t, TraceS2MM, 0)
    aie.runtime_sequence @trace_s2mm_ch0(%arg0: memref<4096xi32>, %n: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, %n, 32][0, 0, 32, 1]) {id = 0 : i64, metadata = @alloc0} : memref<4096xi32>
    }
  }
}

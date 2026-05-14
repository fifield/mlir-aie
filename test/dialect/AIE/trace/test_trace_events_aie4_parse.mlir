//===- test_trace_events_aie4_parse.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

// CHECK-LABEL: module {
module {
  // CHECK: aie.device(npu3)
  aie.device(npu3) {
    // CHECK: %[[CORE:.*]] = aie.tile(0, 2)
    %tile_0_2 = aie.tile(0, 2)
    // CHECK: %[[MEM:.*]] = aie.tile(0, 1)
    %tile_0_1 = aie.tile(0, 1)
    // CHECK: %[[SHIM:.*]] = aie.tile(0, 0)
    %tile_0_0 = aie.tile(0, 0)

    // Core tile with CoreEventAIE4 enums
    // CHECK: aie.trace @core_trace(%[[CORE]]) {
    aie.trace @core_trace(%tile_0_2) {
      // CHECK: aie.trace.event <CoreEventAIE4::PERF_CNT_0>
      aie.trace.event<CoreEventAIE4::PERF_CNT_0>
      // CHECK: aie.trace.event <CoreEventAIE4::BROADCAST_0>
      aie.trace.event<CoreEventAIE4::BROADCAST_0>
      // CHECK: aie.trace.event <CoreEventAIE4::DMA_S2MM_0_START_TASK>
      aie.trace.event<CoreEventAIE4::DMA_S2MM_0_START_TASK>
    }

    // Mem tile with MemTileEventAIE4 enums
    // CHECK: aie.trace @memtile_trace(%[[MEM]]) {
    aie.trace @memtile_trace(%tile_0_1) {
      // CHECK: aie.trace.event <MemTileEventAIE4::BROADCAST_0>
      aie.trace.event<MemTileEventAIE4::BROADCAST_0>
      // CHECK: aie.trace.event <MemTileEventAIE4::CONFLICT_DM_BANK_0>
      aie.trace.event<MemTileEventAIE4::CONFLICT_DM_BANK_0>
    }

    // Shim tile with ShimTileEventAIE4 enums
    // CHECK: aie.trace @shim_trace(%[[SHIM]]) {
    aie.trace @shim_trace(%tile_0_0) {
      // CHECK: aie.trace.event <ShimTileEventAIE4::DMA_S2MM_0_START_TASK>
      aie.trace.event<ShimTileEventAIE4::DMA_S2MM_0_START_TASK>
      // CHECK: aie.trace.event <ShimTileEventAIE4::USER_EVENT_0>
      aie.trace.event<ShimTileEventAIE4::USER_EVENT_0>
    }

    // String-based events also work on AIE4
    // CHECK: aie.trace @string_trace(%[[CORE]]) {
    aie.trace @string_trace(%tile_0_2) {
      // CHECK: aie.trace.event <"PERF_CNT_0">
      aie.trace.event<"PERF_CNT_0">
    }
  }
}

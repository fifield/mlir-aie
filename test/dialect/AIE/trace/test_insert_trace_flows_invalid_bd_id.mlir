//===- test_insert_trace_flows_invalid_bd_id.mlir ---------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// `default-bd-id` names a channel-local slot. The AIE4 TraceS2MM pool
// contains four slots, so 4 and larger are configuration errors rather than
// requests to be silently remapped into another slot.

// RUN: aie-opt %s --aie-insert-trace-flows="default-bd-id=4" -verify-diagnostics

module {
  // expected-error@+1 {{trace buffer descriptor ID 4 is outside the channel-local range}}
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.trace @core_trace(%tile) {
      aie.trace.packet id = 10 type = core
      aie.trace.event <"INSTR_EVENT_0">
      aie.trace.start broadcast = 3
      aie.trace.stop broadcast = 4
    }
    aie.runtime_sequence(%arg0: memref<8xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @core_trace
    }
  }
}

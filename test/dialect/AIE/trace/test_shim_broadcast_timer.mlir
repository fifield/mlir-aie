// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --aie-insert-trace-flows | FileCheck %s

// The traced shim is distinct from the egress shim, so its timer needs syncing.
// BROADCAST_3 is event 235; the reset-event field starts at bit 8.
// CHECK: %[[RESET:.*]] = arith.constant 60160 : i32
// CHECK: %[[ADDR:.*]] = arith.constant 16528 : i32
// CHECK: aiex.npu.write32(%[[ADDR]], %[[RESET]]) {column = 0 : i32, row = 0 : i32}
module {
  aie.device(npu3) {
    %source = aie.tile(0, 0)
    %sink = aie.tile(1, 0)
    aie.trace @shim_trace(%source) {
      aie.trace.packet id=1 type="shimtile"
      aie.trace.event<"USER_EVENT_0">
      aie.trace.start broadcast=3
      aie.trace.stop broadcast=4
    }
    aie.runtime_sequence() {
      aie.trace.host_config {buffer_size = 8192 : i32, egress_shim_col = 1 : i32}
      aie.trace.start_config @shim_trace
    }
  }
}

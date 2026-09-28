// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --aie-insert-trace-flows="shim-channel=1" --verify-diagnostics

module {
  // expected-error@+1 {{trace shim channel 1 is outside the range [0, 1) for TraceS2MM}}
  aie.device(npu3_1col) {
    %core = aie.tile(0, 2)
    aie.trace @trace(%core) {
      aie.trace.event <"INSTR_EVENT_0">
      aie.trace.start broadcast = 3
      aie.trace.stop broadcast = 4
    }
    aie.runtime_sequence() {
      aie.trace.host_config {buffer_size = 8192 : i32}
    }
  }
}

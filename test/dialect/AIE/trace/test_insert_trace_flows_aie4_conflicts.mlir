// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --split-input-file --aie-insert-trace-flows --verify-diagnostics

// Ordinary S2MM claims do not conflict, but an existing dedicated trace route
// must not be silently shared with the trace overlay's buffer program.
module {
  // expected-error@+1 {{no TraceS2MM channels available on shim tile at column 0}}
  aie.device(npu3_1col) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.packet_flow(10) {
      aie.packet_source<%mem, Trace32 : 0>
      aie.packet_dest<%shim, DMATrace : 0>
    }
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

// -----

module {
  // expected-error@+1 {{no TraceS2MM channels available on shim tile at column 0}}
  aie.device(npu3_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.shim_dma_allocation @existing_trace(%shim, TraceS2MM, 0)
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

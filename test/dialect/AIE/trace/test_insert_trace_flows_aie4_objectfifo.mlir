// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --aie-insert-trace-flows | FileCheck %s
// RUN: aie-opt %s --aie-insert-trace-flows="distribute-channels=true" | FileCheck %s

// Object FIFOs reserve ordinary S2MM channels before they lower to flows.
// Neither a pinned nor an unpinned output consumes the dedicated trace DMA.
// A one-column device prevents masking a conflict by redirecting the trace.
// CHECK: %[[SHIM:.*]] = aie.tile(0, 0)
// CHECK: aiex.npu.writebd {bd_id = 3 : i32,
// CHECK-SAME: channel = 0 : i32,
// CHECK-SAME: direction = 3 : i32,
// CHECK: aie.packet_dest<%[[SHIM]], DMATrace : 0>

module {
  aie.device(npu3_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @out0(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out1(%core, {%shim}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
    aie.trace @trace(%core) {
      aie.trace.packet id = 1 type = core
      aie.trace.event <"INSTR_EVENT_0">
      aie.trace.start broadcast = 3
      aie.trace.stop broadcast = 4
    }
    aie.runtime_sequence(%out: memref<16xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @trace
    }
  }
}

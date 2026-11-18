//===- npu3_unsupported_runtime.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Runtime-sequence lowerings that need an AIE4 BD layout or AIE4 trace
// programming refuse AIE4 with a diagnostic.

// RUN: not aie-opt --split-input-file --aie-dma-tasks-to-npu %s 2>&1 | FileCheck %s --check-prefix=TASK
// TASK: error: 'aie.dma_bd' op has no buffer descriptor layout on this tile

// RUN: not aie-opt --split-input-file --aie-dma-to-npu %s 2>&1 | FileCheck %s --check-prefix=MEMCPY
// MEMCPY: error: 'aiex.npu.dma_memcpy_nd' op has no buffer descriptor layout on this tile

// RUN: not aie-opt --split-input-file --aie-insert-trace-flows %s 2>&1 | FileCheck %s --check-prefix=TRACE
// TRACE: error: 'aie.trace' op trace is not yet supported on AIE4

// RUN: not aie-opt --split-input-file --aie-trace-to-config %s 2>&1 | FileCheck %s --check-prefix=TRACECFG
// TRACECFG: error: 'aie.trace' op trace is not yet supported on AIE4

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.runtime_sequence @seq(%arg0 : memref<256xi32>) {
    %task = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
      aie.end
    }
    aiex.dma_start_task(%task)
  }
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @in(%shim, MM2S, 0)
  aie.runtime_sequence @seq(%arg0 : memref<256xi32>) {
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, metadata = @in} : memref<256xi32>
  }
}

// -----

aie.device(npu3) {
  %core = aie.tile(0, 2)
  %shim = aie.tile(0, 0)
  aie.trace @core_trace(%core) {
    aie.trace.mode "Event-Time"
    aie.trace.packet id=1 type=core
    aie.trace.event<"INSTR_EVENT_0">
  }
  aie.runtime_sequence(%arg0: memref<16xi32>) {
    aie.trace.host_config {buffer_size = 1024 : i32}
    aie.trace.start_config @core_trace
  }
}

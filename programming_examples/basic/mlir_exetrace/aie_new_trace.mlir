//===- aie_exec_trace.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// EXECUTION-TRACE MODE DEMO
//
// This demonstrates Execution-Trace mode which traces program control flow:
// - Branch taken/not taken (E_atom/N_atom frames)
// - Indirect branches with new PC (New_PC frames)
// - Loop counter updates (LC frames)
// - Start/Stop frames with PC addresses
//
// Uses the declarative trace API:
// - aie.trace for configuration
// - aie.trace.host_config in runtime sequence for buffer setup
// - aie.trace.start_config in runtime sequence for activation
//
// Trace flows and buffer descriptors are generated automatically
// by the compiler trace lowering pipeline.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    // External kernel function declaration
    func.func private @vector_scalar_mul_aie_scalar(memref<1024xi32>, memref<1024xi32>, memref<1xi32>, i32) attributes {link_with = "scale.o"}

    // Tile declarations
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // ObjectFIFOs for data movement
    aie.objectfifo @in(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @infactor(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1xi32>>
    aie.objectfifo @out(%tile_0_2, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>

    // Core computation
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @infactor(Consume, 1) : !aie.objectfifosubview<memref<1xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
        %c0_0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4 step %c1_1 {
          %2 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<1024xi32>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
          %4 = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @vector_scalar_mul_aie_scalar(%5, %3, %1, %c1024_i32) : (memref<1024xi32>, memref<1024xi32>, memref<1xi32>, i32) -> ()
          aie.objectfifo.release @in(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        }
        aie.objectfifo.release @infactor(Consume, 1)
      }
      aie.end
    }

    // ========================================================================
    // TRACE CONFIGURATION - EXECUTION-TRACE MODE
    // ========================================================================

    // Execution-Trace mode: Traces program control flow (branches, loops, PC changes)
    aie.trace @exec_trace(%tile_0_2) {
      aie.trace.mode "Execution"

      aie.trace.packet id=1 type=core

      // Execution trace doesn't require explicit event configuration.
      // It automatically captures E_atom, N_atom, New_PC, and LC frames.
      aie.trace.event<"INSTR_EVENT_0">

      // Always-on trace (no gating via broadcast)
      aie.trace.start event=<"TRUE">
      aie.trace.stop event=<"NONE">
    }

    // Shim tile trace (Event-Time mode, default)
    aie.trace @shim_trace(%shim_noc_tile_0_0) {
      aie.trace.packet id=2 type=shimtile

      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.event<"DMA_S2MM_1_START_TASK">
      aie.trace.event<"DMA_MM2S_0_START_TASK">
      aie.trace.event<"DMA_S2MM_0_FINISHED_TASK">
      aie.trace.event<"DMA_S2MM_1_FINISHED_TASK">
      aie.trace.event<"DMA_MM2S_0_FINISHED_TASK">
      aie.trace.event<"DMA_S2MM_0_STREAM_STARVATION">
      aie.trace.event<"DMA_S2MM_1_STREAM_STARVATION">

      aie.trace.start event=<"TRUE">
      aie.trace.stop event=<"NONE">
    }

    // ========================================================================
    // RUNTIME SEQUENCE WITH TRACE ACTIVATION
    // ========================================================================

    aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<1xi32>, %arg2: memref<4096xi32>) {

      // Configure trace output buffer (8 KB)
      aie.trace.host_config buffer_size = 8192

      // Start trace configurations
      aie.trace.start_config @shim_trace
      aie.trace.start_config @exec_trace

      // Configure DMA tasks for input, factor, and output
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xi32>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}

      %1 = aiex.dma_configure_task_for @infactor {
        aie.dma_bd(%arg1 : memref<1xi32>, 0, 1, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}

      %2 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg2 : memref<4096xi32>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}

      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%0)
      aiex.dma_await_task(%1)
      aiex.dma_await_task(%2)
    }
  }
}

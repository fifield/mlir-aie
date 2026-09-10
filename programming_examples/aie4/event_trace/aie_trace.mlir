//===- aie_trace.mlir ------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Ported from basic/event_trace to AIE4 (npu3 target).
//
// This example uses:
// - aie.trace operation for declarative trace configuration
// - aie.trace.event for specifying events to capture
// - aie.trace.start_config in runtime sequence
//
// The passes aie-trace-to-config and aie-inline-trace-config will lower this.
//
// AIE4 differences from AIE2:
// - Core trace: Trace0_Control0/1, Trace0_Event0/1 (and Trace1_ for mem events)
// - Port field: Port_N_Manager_Subordinate (instead of Port_N_Master_Slave)
// - Shim trace: Trace_A_Control0/1, Trace_A_Event0/1
// - Timer: core=0xA0110, shim=Timer_A_Control=0x4090
// - Broadcast: shim Event_Broadcast_A_N at 0x40B0+N*4
// - DMA S2MM ch1 BDs at 0x58000+ (stride 0x30), BD0_1 (address field) = 0x58004
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu3) {
    // External kernel function declaration
    func.func private @vector_scalar_mul_aie_scalar(memref<1024xi32>, memref<1024xi32>, memref<1xi32>, i32) attributes {link_with = "project/scale.o"}

    // Tile declarations
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %tile_0_2 = aie.tile(0, 2)

    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_0_0, Control32 : 0>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }

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
    // TRACE CONFIGURATION
    // ========================================================================

    // Trace configuration for compute tile (0,2) - core events (Trace0 unit)
    aie.trace @core_trace(%tile_0_2) {
      // Set trace mode (Event-Time captures timestamps)
      aie.trace.mode "Event-Time"

      // Configure packet routing (ID and type for packet-switched routing)
      aie.trace.packet id=1 type=core

      // Specify which events to capture (up to 8 events)
      aie.trace.event<"INSTR_EVENT_0">        // User event 0 (start marker)
      aie.trace.event<"INSTR_EVENT_1">        // User event 1 (end marker)
      aie.trace.event<"INSTR_VECTOR">         // Vector instructions
      aie.trace.event<"MEMORY_STALL">         // Memory access stalls
      aie.trace.event<"STREAM_STALL">         // Stream buffer stalls
      aie.trace.event<"LOCK_STALL">           // Lock acquisition stalls
      aie.trace.event<"DMA_S2MM_0_RUNNING">   // DMA S2MM ch0 active
      aie.trace.event<"DMA_MM2S_0_RUNNING">   // DMA MM2S ch0 active
      // Note: aie.trace.port (PORT_RUNNING/IDLE events) requires
      // getStreamSwitchPortIndex support for AIE4, not yet implemented.

      // Specify start/stop control (broadcast events)
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    // Trace configuration for compute tile (0,2) - DMA/memory events (Trace1 unit)
    // In AIE4, there is no separate memory module; the second trace unit (Trace1)
    // on the core tile is used to capture DMA events instead.
    aie.trace @mem_trace(%tile_0_2) {
      // Set trace mode (Event-Time captures timestamps)
      aie.trace.mode "Event-Time"

      // Configure packet routing (ID and type for packet-switched routing)
      aie.trace.packet id=3 type=mem

      // Specify which events to capture (up to 8 events)
      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.event<"DMA_S2MM_1_START_TASK">
      aie.trace.event<"DMA_MM2S_0_START_TASK">
      aie.trace.event<"DMA_S2MM_0_FINISHED_TASK">
      aie.trace.event<"DMA_S2MM_1_FINISHED_TASK">
      aie.trace.event<"DMA_MM2S_0_FINISHED_TASK">
      aie.trace.event<"DMA_S2MM_0_STREAM_STARVATION">
      aie.trace.event<"DMA_S2MM_1_STREAM_STARVATION">

      // Specify start/stop control (broadcast events)
      aie.trace.start event=<"BROADCAST_15">
      aie.trace.stop event=<"BROADCAST_14">
    }

    // Trace configuration for shim tile (0,0)
    // Captures DMA activity at the interface to DDR
    aie.trace @shim_trace(%shim_noc_tile_0_0) {
      aie.trace.packet id=2 type=shimtile

      // Shim DMA events
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

    // Runtime sequence: 3 data args + trace buffer arg
    aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<1xi32>, %arg2: memref<4096xi32>, %arg3: memref<8192xi32>) {

      // ========================================================================
      // TRACE INITIALIZATION
      // ========================================================================

      // Configure trace output buffer (8192 bytes)
      // arg_idx=3: trace buffer is the 4th XRT kernel argument (after in, factor, out)
      aie.trace.host_config buffer_size = 8192 arg_idx = 3

      // Declarative trace configuration is lowered by passes
      aie.trace.start_config @core_trace
      aie.trace.start_config @mem_trace
      aie.trace.start_config @shim_trace

      // ========================================================================
      // DATA TRANSFER CONFIGURATION
      // ========================================================================

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

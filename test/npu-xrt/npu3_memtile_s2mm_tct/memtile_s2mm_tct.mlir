//===- memtile_s2mm_tct.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NPU3 hardware regression for the MEM-tile S2MM TCT actor mapping across
// multiple channels and columns.
//
// An AIE4 MEM tile has eight S2MM channels, and MEM S2MMn uses TCT actor n.
//
// Each dma_await_task lowers through aiex.npu.sync to aiex.cert.wait_tcts. The
// chan2actor_mem_s2mm table must map channels 0, 1, 2, and 7 to actors 0, 1,
// 2, and 7. The runtime job executes on the controller at shim (0,0), so every
// MEM tile's TCT route must terminate there even when the DMA is in another
// column. S2MM7 belongs to the upper channel bank and emits its TCT through
// the second control-packet handler response port, Control32:1.
//
// This is the minimal hardware counterpart of
// test/unit_tests/aie4_cert/12_memtile_dma_configure_task_token, which is
// restricted to mesimulator.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu3) {
    %shim_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
    %shim_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
    %shim_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}

    %mem_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
    %mem_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
    %mem_2_1 = aie.tile(2, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}

    %mem_buffer_0 = aie.buffer(%mem_0_1) {sym_name = "mem_buffer_0"} : memref<64xi32>
    %mem_buffer_1 = aie.buffer(%mem_1_1) {sym_name = "mem_buffer_1"} : memref<64xi32>
    %mem_buffer_2 = aie.buffer(%mem_2_1) {sym_name = "mem_buffer_2"} : memref<64xi32>
    %mem_buffer_7 = aie.buffer(%mem_2_1) {sym_name = "mem_buffer_7"} : memref<64xi32>

    // Three independent host-input segments enter different MEM S2MM actors.
    aie.flow(%shim_0_0, DMA : 0, %mem_0_1, DMA : 0)
    aie.flow(%shim_1_0, DMA : 0, %mem_1_1, DMA : 1)
    aie.flow(%shim_2_0, DMA : 0, %mem_2_1, DMA : 2)
    aie.flow(%shim_2_0, DMA : 1, %mem_2_1, DMA : 7)

    // The awaits execute on shim (0,0), so all three MEM-tile TCT routes must
    // return to that controller, including the routes originating in columns
    // 1 and 2.
    aie.packet_flow(2) {
      aie.packet_source<%mem_0_1, Control32 : 0>
      aie.packet_dest<%shim_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(3) {
      aie.packet_source<%mem_1_1, Control32 : 0>
      aie.packet_dest<%shim_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(4) {
      aie.packet_source<%mem_2_1, Control32 : 0>
      aie.packet_source<%mem_2_1, Control32 : 1>
      aie.packet_dest<%shim_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}

    aie.runtime_sequence @sequence(%input: memref<256xi32>) {
      %shim_mm2s_0 = aiex.dma_configure_task(%shim_0_0, MM2S, 0) {
        aie.dma_bd(%input : memref<256xi32> offset = 0 len = 64) {bd_id = 0 : i32}
        aie.end
      }
      %mem_s2mm_0 = aiex.dma_configure_task(%mem_0_1, S2MM, 0) {
        aie.dma_bd(%mem_buffer_0 : memref<64xi32> offset = 0 len = 64) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}

      %shim_mm2s_1 = aiex.dma_configure_task(%shim_1_0, MM2S, 0) {
        aie.dma_bd(%input : memref<256xi32> offset = 64 len = 64) {bd_id = 0 : i32}
        aie.end
      }
      %mem_s2mm_1 = aiex.dma_configure_task(%mem_1_1, S2MM, 1) {
        aie.dma_bd(%mem_buffer_1 : memref<64xi32> offset = 0 len = 64) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}

      %shim_mm2s_2 = aiex.dma_configure_task(%shim_2_0, MM2S, 0) {
        aie.dma_bd(%input : memref<256xi32> offset = 128 len = 64) {bd_id = 0 : i32}
        aie.end
      }
      %mem_s2mm_2 = aiex.dma_configure_task(%mem_2_1, S2MM, 2) {
        aie.dma_bd(%mem_buffer_2 : memref<64xi32> offset = 0 len = 64) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}

      %shim_mm2s_3 = aiex.dma_configure_task(%shim_2_0, MM2S, 1) {
        aie.dma_bd(%input : memref<256xi32> offset = 192 len = 64) {bd_id = 1 : i32}
        aie.end
      }
      %mem_s2mm_7 = aiex.dma_configure_task(%mem_2_1, S2MM, 7) {
        aie.dma_bd(%mem_buffer_7 : memref<64xi32> offset = 0 len = 64) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}

      aiex.dma_start_task(%shim_mm2s_0)
      aiex.dma_start_task(%mem_s2mm_0)
      aiex.dma_start_task(%shim_mm2s_1)
      aiex.dma_start_task(%mem_s2mm_1)
      aiex.dma_start_task(%shim_mm2s_2)
      aiex.dma_start_task(%mem_s2mm_2)
      aiex.dma_start_task(%shim_mm2s_3)
      aiex.dma_start_task(%mem_s2mm_7)

      aiex.dma_await_task(%mem_s2mm_7)
      aiex.dma_await_task(%mem_s2mm_2)
      aiex.dma_await_task(%mem_s2mm_1)
      aiex.dma_await_task(%mem_s2mm_0)
    }
  }
}

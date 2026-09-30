//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stream 0 supplies 48 words through one 1024-bit and one 512-bit read.
// Stream 1 supplies 16 words through a 512-bit read. Return all 64 words
// in that order through an objectfifo drained by the shim.
module {
  aie.device(npu3) {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %tile_0_2 = aie.tile(0, 2)

    aie.packet_flow(0) {
      aie.packet_source<%tile_0_0, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}

    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, Core : 0)
    aie.flow(%tile_0_0, DMA : 1, %tile_0_2, Core : 1)
    aie.shim_dma_allocation @in0(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @in1(%tile_0_0, MM2S, 1)

    aie.objectfifo @out(%tile_0_2, {%tile_0_0}, 1 : i32) : !aie.objectfifo<memref<64xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c48 = arith.constant 48 : index
      %s0 = arith.constant 0 : i32
      %s1 = arith.constant 1 : i32
      %elem = aie.objectfifo.acquire @out(Produce, 1) : memref<64xi32>
      %a = aie.get_stream(%s0 : i32) : vector<32xi32>
      vector.store %a, %elem[%c0] : memref<64xi32>, vector<32xi32>
      %b = aie.get_stream(%s0 : i32) : vector<16xi32>
      vector.store %b, %elem[%c32] : memref<64xi32>, vector<16xi32>
      %c = aie.get_stream(%s1 : i32) : vector<16xi32>
      vector.store %c, %elem[%c48] : memref<64xi32>, vector<16xi32>
      aie.objectfifo.release @out(Produce, 1)
      aie.end
    }

    aie.runtime_sequence @sequence(%in0 : memref<48xi32>, %in1 : memref<16xi32>, %out : memref<64xi32>) {
      %tout = aiex.dma_configure_task_for @out {
        aie.dma_bd(%out : memref<64xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      %t0 = aiex.dma_configure_task_for @in0 {
        aie.dma_bd(%in0 : memref<48xi32> offset = 0 len = 48)
        aie.end
      }
      %t1 = aiex.dma_configure_task_for @in1 {
        aie.dma_bd(%in1 : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      aiex.dma_start_task(%tout)
      aiex.dma_start_task(%t0)
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%tout)
      aiex.dma_free_task(%t0)
      aiex.dma_free_task(%t1)
    }
  }
}

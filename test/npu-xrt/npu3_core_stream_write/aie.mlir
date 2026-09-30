//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mix scalar, 256-bit, and 512-bit writes. The first vector write crosses
// an output buffer boundary; subsequent writes complete all four 512-bit
// transfers without TLAST.
module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %tile = aie.tile(0, 2)

    aie.packet_flow(0) {
      aie.packet_source<%shim, Control32 : 0>
      aie.packet_dest<%shim, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}

    aie.objectfifo @in(%shim, {%tile}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.flow(%tile, Core : 0, %shim, DMA : 0)
    aie.shim_dma_allocation @out(%shim, S2MM, 0)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c48 = arith.constant 48 : index
      %c56 = arith.constant 56 : index
      %s0 = arith.constant 0 : i32
      %elem = aie.objectfifo.acquire @in(Consume, 1) : memref<64xi32>
      %first8 = vector.load %elem[%c0] : memref<64xi32>, vector<8xi32>
      %word0 = vector.extract %first8[0] : i32 from vector<8xi32>
      aie.put_stream(%s0 : i32, %word0 : i32)
      %word1 = vector.extract %first8[1] : i32 from vector<8xi32>
      aie.put_stream(%s0 : i32, %word1 : i32)
      %word2 = vector.extract %first8[2] : i32 from vector<8xi32>
      aie.put_stream(%s0 : i32, %word2 : i32)
      %word3 = vector.extract %first8[3] : i32 from vector<8xi32>
      aie.put_stream(%s0 : i32, %word3 : i32)
      %word4 = vector.extract %first8[4] : i32 from vector<8xi32>
      aie.put_stream(%s0 : i32, %word4 : i32)
      %word5 = vector.extract %first8[5] : i32 from vector<8xi32>
      aie.put_stream(%s0 : i32, %word5 : i32)
      %word6 = vector.extract %first8[6] : i32 from vector<8xi32>
      aie.put_stream(%s0 : i32, %word6 : i32)
      %word7 = vector.extract %first8[7] : i32 from vector<8xi32>
      aie.put_stream(%s0 : i32, %word7 : i32)
      %v16 = vector.load %elem[%c16] : memref<64xi32>, vector<16xi32>
      aie.put_stream(%s0 : i32, %v16 : vector<16xi32>)
      %v8 = vector.load %elem[%c8] : memref<64xi32>, vector<8xi32>
      aie.put_stream(%s0 : i32, %v8 : vector<8xi32>)
      %v32 = vector.load %elem[%c32] : memref<64xi32>, vector<16xi32>
      aie.put_stream(%s0 : i32, %v32 : vector<16xi32>)
      %v48 = vector.load %elem[%c48] : memref<64xi32>, vector<8xi32>
      aie.put_stream(%s0 : i32, %v48 : vector<8xi32>)
      %v56 = vector.load %elem[%c56] : memref<64xi32>, vector<8xi32>
      aie.put_stream(%s0 : i32, %v56 : vector<8xi32>)
      aie.objectfifo.release @in(Consume, 1)
      aie.end
    }

    aie.runtime_sequence @sequence(%in : memref<64xi32>, %out : memref<64xi32>) {
      %tout = aiex.dma_configure_task_for @out {
        aie.dma_bd(%out : memref<64xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      %tin = aiex.dma_configure_task_for @in {
        aie.dma_bd(%in : memref<64xi32> offset = 0 len = 64)
        aie.end
      }
      aiex.dma_start_task(%tout)
      aiex.dma_start_task(%tin)
      aiex.dma_await_task(%tout)
      aiex.dma_free_task(%tin)
    }
  }
}

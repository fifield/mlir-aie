//===- aie4.mlir -----------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices Inc.
//
// REQUIRES: mesimulator

// RUN: mkdir -p project0/
// RUN: cp %aie_obj_root/third_party/cert/main/cert_medusa.elf project0/cert_medusa.elf
// RUN: aiecc --tmpdir=project0 --xchesscc --xbridge --aie-generate-txn --txn-name=transaction.bin %S/base.mlir
// RUN: aiecc --tmpdir=project1 --xchesscc --xbridge --no-compile-host --generate-full-elf --full-elf-name=aie4.elf --aiesim %s %test_utils_flags %S/test.cpp
// RUN: project1/mesim.sh | FileCheck --check-prefix=MESIM %s
//
// MESIM: device model: {{.*}}aie4
// MESIM: PASS!
//
// Test: AIE4 shim DMA 4D (D3) addressing via aiex.npu.dma_memcpy_nd.
//
// This is the npu.dma_memcpy_nd counterpart of 16_shim_dma_4d (which uses the
// aiex.dma_configure_task task path). Here the two shim/DDR transfers are
// expressed with aiex.npu.dma_memcpy_nd in the runtime sequence instead.
//
// The shim MM2S read uses a genuine four-data-dimension (D0..D3) access pattern
// with NO repeat -- exactly the addressing the AIE4 shim-D3 extension adds. The
// memcpy_nd access pattern is outermost-first [iteration, d3, d2, d1, d0]; the
// leading iteration entry is size 1 (no repeat), so all motion comes from the
// four data dimensions:
//
//   for i3 in 0..1 (d3, stride 16)
//    for i2 in 0..1 (d2, stride 4)
//     for i1 in 0..1 (d1, stride 8)
//      for i0 in 0..3 (d0, stride 1)         // d0 contiguous (AIE4 rule)
//        addr = 16*i3 + 4*i2 + 8*i1 + i0
//
// The strides {16,4,8,1} make the pattern a bijection of [0,32) (so the output
// is a pure permutation of the input) and irreducible to fewer than four data
// dimensions: no two adjacent loop levels can merge.
//
// Data path: shim MM2S (4D scatter, memcpy_nd) -> memtile S2MM -> memtile MM2S
// -> shim S2MM (contiguous, memcpy_nd) -> DDR output.
//
module {
  aie.device(npu3) {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}

    %in_buff = aie.buffer(%tile_0_1) {sym_name = "in_buff"} : memref<32xi32>

    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)

    // Shim DMA allocations referenced by npu.dma_memcpy_nd metadata.
    aie.shim_dma_allocation @in_alloc(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out_alloc(%tile_0_0, S2MM, 0)

    // Packet flows for issue_token support
    aie.packet_flow(2) {
      aie.packet_source<%tile_0_1, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}

    aie.packet_flow(4) {
      aie.packet_source<%tile_0_0, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}

    aie.runtime_sequence(%arg0: memref<32xi32>, %arg1: memref<32xi32>) {
      // Shim MM2S: read DDR input with a four-data-dimension (D0..D3) scatter.
      // Access pattern order is [iteration, d3, d2, d1, d0]; iteration size 1
      // means no repeat, so this exercises D3 as a pure data dimension.
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0, 0][1, 2, 2, 2, 4][0, 16, 4, 8, 1]) {
        metadata = @in_alloc, id = 0 : i64
      } : memref<32xi32>

      // Memtile S2MM: receive the 32 streamed elements contiguously.
      %t1 = aiex.dma_configure_task(%tile_0_1, S2MM, 0) {
        aie.dma_bd(%in_buff : memref<32xi32>, 0, 32) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)

      // Memtile MM2S: send the buffer back contiguously.
      %t2 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%in_buff : memref<32xi32>, 0, 32) {bd_id = 1 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)

      // Shim S2MM: write to DDR output contiguously, then wait for completion.
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0, 0][1, 1, 1, 1, 32][0, 0, 0, 0, 1]) {
        metadata = @out_alloc, id = 1 : i64
      } : memref<32xi32>
      aiex.npu.dma_wait {symbol = @out_alloc}
    }
  }
}

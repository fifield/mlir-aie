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
// RUN: aiecc --tmpdir=project1 --xchesscc --xbridge --no-compile-host --get-full-elf --full-elf-name=aie4.elf --aiesim %s %test_utils_flags %S/test.cpp
// RUN: project1/mesim.sh | FileCheck --check-prefix=MESIM %s
//
// MESIM: device model: {{.*}}aie4
// MESIM: PASS!
//
// Test: AIE4 shim DMA 4D (D3) addressing, end-to-end.
//
// This exercises the AIE4 shim-D3 extension: a shim DMA buffer descriptor that
// uses a full 5-entry access pattern [iteration, d3, d2, d1, d0].
//
// The shim MM2S below reads the 64-element DDR input in a 5-dimensional,
// fully-scattered order (a bijection of [0,64), so it is a pure permutation):
//
//   for it in 0..1 (iteration, stride 16)
//    for i3 in 0..1 (d3, stride 4)
//     for i2 in 0..1 (d2, stride 32)
//      for i1 in 0..1 (d1, stride 8)
//       for i0 in 0..3 (d0, stride 1)         // d0 is contiguous (AIE4 rule)
//         addr = 16*it + 4*i3 + 32*i2 + 8*i1 + i0
//
// The strides {16,4,32,8,1} are chosen so that no two adjacent loop levels can
// be merged (a merge of an outer level O over inner level I requires
// stride_O == size_I*stride_I, which holds for none of the pairs here). The
// pattern is thus irreducibly 5-dimensional and impossible to encode in the
// pre-D3 four-entry shim BD.
//
// Data path: shim MM2S (4D scatter) -> memtile S2MM (contiguous) -> memtile
// MM2S (contiguous) -> shim S2MM (contiguous) -> DDR output. The host checks
// that the output is the expected permutation of the input.
//
module {
  aie.device(npu3) {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}

    %in_buff = aie.buffer(%tile_0_1) {sym_name = "in_buff"} : memref<64xi32>

    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)

    // Packet flows for issue_token support
    aie.packet_flow(2) {
      aie.packet_source<%tile_0_1, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}

    aie.packet_flow(4) {
      aie.packet_source<%tile_0_0, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}

    aie.runtime_sequence(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
      // Shim MM2S: read DDR input with a 5-entry (4 data dims + iteration)
      // access pattern. Order is outermost-first: [iteration, d3, d2, d1, d0].
      // buffer_length 32 = product of the four data dims (4*2*2*2); the
      // iteration dimension (size 2, stride 16) repeats the whole BD, moving
      // 64 elements total. This BD is only legal with the AIE4 shim-D3
      // extension.
      //
      // repeat_count = 1 makes the queue push re-run the BD once (2 iterations
      // total). The BD's iteration_size/iteration_stride supply the per-repeat
      // address advance; the push repeat_count supplies the loop count. Both
      // are required and together they match exactly what aiex.npu.dma_memcpy_nd
      // emits for the same 5-dim access pattern.
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<64xi32>, 0, 32,
                   [<size = 2, stride = 16>,
                    <size = 2, stride = 4>,
                    <size = 2, stride = 32>,
                    <size = 2, stride = 8>,
                    <size = 4, stride = 1>]) {bd_id = 0 : i32}
        aie.end
      } {repeat_count = 1 : i32}

      // Memtile S2MM: receive the 64 streamed elements contiguously.
      %t1 = aiex.dma_configure_task(%tile_0_1, S2MM, 0) {
        aie.dma_bd(%in_buff : memref<64xi32>, 0, 64) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}

      aiex.dma_start_task(%t0)
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)

      // Memtile MM2S: send the buffer back contiguously.
      %t2 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%in_buff : memref<64xi32>, 0, 64) {bd_id = 1 : i32}
        aie.end
      }

      // Shim S2MM: write to DDR output contiguously.
      %t3 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%arg1 : memref<64xi32>, 0, 64) {bd_id = 1 : i32}
        aie.end
      } {issue_token = true}

      aiex.dma_start_task(%t2)
      aiex.dma_start_task(%t3)
      aiex.dma_await_task(%t3)
    }
  }
}

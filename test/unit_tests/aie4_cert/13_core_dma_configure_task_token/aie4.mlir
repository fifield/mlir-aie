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
// RUN: aiecc.py --tmpdir=project0 --xchesscc --xbridge --aie-generate-txn --txn-name=transaction.bin %S/base.mlir
// RUN: aiecc.py --tmpdir=project1 --xchesscc --xbridge --no-compile-host --generate-full-elf --full-elf-name=aie4.elf --aiesim %s %test_utils_flags %S/test.cpp
// RUN: project1/mesim.sh | FileCheck --check-prefix=MESIM %s
//
// MESIM: device model: {{.*}}aie4
// MESIM: PASS!
//
module {
  aie.device(npu3) {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}

    // Core tile buffers
    %in_buff = aie.buffer(%tile_0_2) {sym_name = "in_buff"} : memref<1024xi32>
    // %out_buff = aie.buffer(%tile_0_2) {sym_name = "out_buff"} : memref<1024xi32>

    // Flows: shim -> core -> shim
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)

    // Packet flows for issue_token support
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}

    aie.packet_flow(3) {
      aie.packet_source<%tile_0_0, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}

    %lock_0_0 = aie.lock(%tile_0_2) {initial_value = 0 : i32}
    // Core program to add 1 to each element
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1024 = arith.constant 1024 : index
      %c1_i32 = arith.constant 1 : i32

      aie.use_lock(%lock_0_0, AcquireGreaterEqual, 0x1)
      scf.for %arg0 = %c0 to %c1024 step %c1 {
        %0 = memref.load %in_buff[%arg0] : memref<1024xi32>
        %1 = arith.addi %0, %c1_i32 : i32
        memref.store %1, %in_buff[%arg0] : memref<1024xi32>
      }
      aie.end
    }

    aie.runtime_sequence(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
      // Configure shim DMA to send data to shim
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      }

      // Configure core DMA to receive from shim
      %t3 = aiex.dma_configure_task(%tile_0_2, S2MM, 0) {
        aie.dma_bd(%in_buff : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}

      // Start input path
      aiex.dma_start_task(%t0)
      aiex.dma_start_task(%t3)
      aiex.dma_await_task(%t3)

      // Configure core DMA to send to shim
      %t4 = aiex.dma_configure_task(%tile_0_2, MM2S, 0) {
        aie.dma_bd(%in_buff : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32}
        aie.end
      }

      // Configure shim DMA to receive from core
      %t7 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%arg1 : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32}
        aie.end
      } {issue_token = true}

      // Start output path
      aiex.dma_start_task(%t4)
      aiex.dma_start_task(%t7)
      aiex.dma_await_task(%t7)
    }
  }
}

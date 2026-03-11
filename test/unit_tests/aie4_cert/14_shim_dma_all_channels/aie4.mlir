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
// MESIM: PASS
//
// Test: AIE4z shim DMA all-channel BD coverage
// Validates that all 8 MM2S channels and all 4 S2MM channels work with
// various BD IDs (0-15) under native AIE4z mode (16 private BDs per channel).
// Each column performs a memtile-bounce loopback: shim->memtile->shim.
//
module {
  aie.device(npu3z) {
    // Shim tiles (col 0-8)
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
    %tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
    %tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
    %tile_4_0 = aie.tile(4, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %tile_5_0 = aie.tile(5, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
    %tile_6_0 = aie.tile(6, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
    %tile_7_0 = aie.tile(7, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}

    // Memtiles (col 0-7, row 1)
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 9>}
    %tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 10>}
    %tile_2_1 = aie.tile(2, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 11>}
    %tile_3_1 = aie.tile(3, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 12>}
    %tile_4_1 = aie.tile(4, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 13>}
    %tile_5_1 = aie.tile(5, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 14>}
    %tile_6_1 = aie.tile(6, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_7_1 = aie.tile(7, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 16>}

    // Memtile buffers (1024xi32 each)
    %buf_0 = aie.buffer(%tile_0_1) {sym_name = "buf_0"} : memref<1024xi32>
    %buf_1 = aie.buffer(%tile_1_1) {sym_name = "buf_1"} : memref<1024xi32>
    %buf_2 = aie.buffer(%tile_2_1) {sym_name = "buf_2"} : memref<1024xi32>
    %buf_3 = aie.buffer(%tile_3_1) {sym_name = "buf_3"} : memref<1024xi32>
    %buf_4 = aie.buffer(%tile_4_1) {sym_name = "buf_4"} : memref<1024xi32>
    %buf_5 = aie.buffer(%tile_5_1) {sym_name = "buf_5"} : memref<1024xi32>
    %buf_6 = aie.buffer(%tile_6_1) {sym_name = "buf_6"} : memref<1024xi32>
    %buf_7 = aie.buffer(%tile_7_1) {sym_name = "buf_7"} : memref<1024xi32>

    // Data flows: shim -> memtile (MM2S) and memtile -> shim (S2MM)
    // Col 0: MM2S ch0, S2MM ch0
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
    // Col 1: MM2S ch1, S2MM ch1
    aie.flow(%tile_1_0, DMA : 1, %tile_1_1, DMA : 0)
    aie.flow(%tile_1_1, DMA : 0, %tile_1_0, DMA : 1)
    // Col 2: MM2S ch2, S2MM ch2
    aie.flow(%tile_2_0, DMA : 2, %tile_2_1, DMA : 0)
    aie.flow(%tile_2_1, DMA : 0, %tile_2_0, DMA : 2)
    // Col 3: MM2S ch3, S2MM ch3
    aie.flow(%tile_3_0, DMA : 3, %tile_3_1, DMA : 0)
    aie.flow(%tile_3_1, DMA : 0, %tile_3_0, DMA : 3)
    // Col 4: MM2S ch4, S2MM ch3
    aie.flow(%tile_4_0, DMA : 4, %tile_4_1, DMA : 0)
    aie.flow(%tile_4_1, DMA : 0, %tile_4_0, DMA : 3)
    // Col 5: MM2S ch5, S2MM ch2
    aie.flow(%tile_5_0, DMA : 5, %tile_5_1, DMA : 0)
    aie.flow(%tile_5_1, DMA : 0, %tile_5_0, DMA : 2)
    // Col 6: MM2S ch6, S2MM ch1
    aie.flow(%tile_6_0, DMA : 6, %tile_6_1, DMA : 0)
    aie.flow(%tile_6_1, DMA : 0, %tile_6_0, DMA : 1)
    // Col 7: MM2S ch7, S2MM ch0
    aie.flow(%tile_7_0, DMA : 7, %tile_7_1, DMA : 0)
    aie.flow(%tile_7_1, DMA : 0, %tile_7_0, DMA : 0)

    // Packet flows for issue_token support (shim tiles)
    aie.packet_flow(0) {
      aie.packet_source<%tile_0_0, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(1) {
      aie.packet_source<%tile_1_0, Control32 : 0>
      aie.packet_source<%tile_1_0, Control32 : 1>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(2) {
      aie.packet_source<%tile_2_0, Control32 : 0>
      aie.packet_source<%tile_2_0, Control32 : 1>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(3) {
      aie.packet_source<%tile_3_0, Control32 : 0>
      aie.packet_source<%tile_3_0, Control32 : 1>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(4) {
      aie.packet_source<%tile_4_0, Control32 : 0>
      aie.packet_source<%tile_4_0, Control32 : 1>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(5) {
      aie.packet_source<%tile_5_0, Control32 : 0>
      aie.packet_source<%tile_5_0, Control32 : 1>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(6) {
      aie.packet_source<%tile_6_0, Control32 : 0>
      aie.packet_source<%tile_6_0, Control32 : 1>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(7) {
      aie.packet_source<%tile_7_0, Control32 : 0>
      aie.packet_source<%tile_7_0, Control32 : 1>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}

    // Packet flows for issue_token support (memtiles)
    aie.packet_flow(9) {
      aie.packet_source<%tile_0_1, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(10) {
      aie.packet_source<%tile_1_1, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(11) {
      aie.packet_source<%tile_2_1, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(12) {
      aie.packet_source<%tile_3_1, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(13) {
      aie.packet_source<%tile_4_1, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(14) {
      aie.packet_source<%tile_5_1, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%tile_6_1, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(16) {
      aie.packet_source<%tile_7_1, Control32 : 0>
      aie.packet_dest<%tile_0_0, Controller32 : 0>
    } {keep_pkt_header = true, priority_route = true}

    aie.runtime_sequence(%arg0: memref<8192xi32>, %arg1: memref<8192xi32>) {
      // Column 0: MM2S ch0, S2MM ch0, BD 0/1
      %t0_0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8192xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      }
      %t0_1 = aiex.dma_configure_task(%tile_0_1, S2MM, 0) {
        aie.dma_bd(%buf_0 : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t0_0)
      aiex.dma_start_task(%t0_1)
      aiex.dma_await_task(%t0_1)
      %t0_2 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%buf_0 : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32}
        aie.end
      }
      %t0_3 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%arg1 : memref<8192xi32>, 0, 1024) {bd_id = 1 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t0_2)
      aiex.dma_start_task(%t0_3)
      aiex.dma_await_task(%t0_3)

      // Column 1: MM2S ch1, S2MM ch1, BD 2/3
      %t1_0 = aiex.dma_configure_task(%tile_1_0, MM2S, 1) {
        aie.dma_bd(%arg0 : memref<8192xi32>, 1024, 1024) {bd_id = 2 : i32}
        aie.end
      }
      %t1_1 = aiex.dma_configure_task(%tile_1_1, S2MM, 0) {
        aie.dma_bd(%buf_1 : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1_0)
      aiex.dma_start_task(%t1_1)
      aiex.dma_await_task(%t1_1)
      %t1_2 = aiex.dma_configure_task(%tile_1_1, MM2S, 0) {
        aie.dma_bd(%buf_1 : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32}
        aie.end
      }
      %t1_3 = aiex.dma_configure_task(%tile_1_0, S2MM, 1) {
        aie.dma_bd(%arg1 : memref<8192xi32>, 1024, 1024) {bd_id = 3 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1_2)
      aiex.dma_start_task(%t1_3)
      aiex.dma_await_task(%t1_3)

      // Column 2: MM2S ch2, S2MM ch2, BD 4/5
      %t2_0 = aiex.dma_configure_task(%tile_2_0, MM2S, 2) {
        aie.dma_bd(%arg0 : memref<8192xi32>, 2048, 1024) {bd_id = 4 : i32}
        aie.end
      }
      %t2_1 = aiex.dma_configure_task(%tile_2_1, S2MM, 0) {
        aie.dma_bd(%buf_2 : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2_0)
      aiex.dma_start_task(%t2_1)
      aiex.dma_await_task(%t2_1)
      %t2_2 = aiex.dma_configure_task(%tile_2_1, MM2S, 0) {
        aie.dma_bd(%buf_2 : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32}
        aie.end
      }
      %t2_3 = aiex.dma_configure_task(%tile_2_0, S2MM, 2) {
        aie.dma_bd(%arg1 : memref<8192xi32>, 2048, 1024) {bd_id = 5 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2_2)
      aiex.dma_start_task(%t2_3)
      aiex.dma_await_task(%t2_3)

      // Column 3: MM2S ch3, S2MM ch3, BD 6/7
      %t3_0 = aiex.dma_configure_task(%tile_3_0, MM2S, 3) {
        aie.dma_bd(%arg0 : memref<8192xi32>, 3072, 1024) {bd_id = 6 : i32}
        aie.end
      }
      %t3_1 = aiex.dma_configure_task(%tile_3_1, S2MM, 0) {
        aie.dma_bd(%buf_3 : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t3_0)
      aiex.dma_start_task(%t3_1)
      aiex.dma_await_task(%t3_1)
      %t3_2 = aiex.dma_configure_task(%tile_3_1, MM2S, 0) {
        aie.dma_bd(%buf_3 : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32}
        aie.end
      }
      %t3_3 = aiex.dma_configure_task(%tile_3_0, S2MM, 3) {
        aie.dma_bd(%arg1 : memref<8192xi32>, 3072, 1024) {bd_id = 7 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t3_2)
      aiex.dma_start_task(%t3_3)
      aiex.dma_await_task(%t3_3)

      // Column 4: MM2S ch4, S2MM ch3, BD 8/9
      %t4_0 = aiex.dma_configure_task(%tile_4_0, MM2S, 4) {
        aie.dma_bd(%arg0 : memref<8192xi32>, 4096, 1024) {bd_id = 8 : i32}
        aie.end
      }
      %t4_1 = aiex.dma_configure_task(%tile_4_1, S2MM, 0) {
        aie.dma_bd(%buf_4 : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t4_0)
      aiex.dma_start_task(%t4_1)
      aiex.dma_await_task(%t4_1)
      %t4_2 = aiex.dma_configure_task(%tile_4_1, MM2S, 0) {
        aie.dma_bd(%buf_4 : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32}
        aie.end
      }
      %t4_3 = aiex.dma_configure_task(%tile_4_0, S2MM, 3) {
        aie.dma_bd(%arg1 : memref<8192xi32>, 4096, 1024) {bd_id = 9 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t4_2)
      aiex.dma_start_task(%t4_3)
      aiex.dma_await_task(%t4_3)

      // Column 5: MM2S ch5, S2MM ch2, BD 10/11
      %t5_0 = aiex.dma_configure_task(%tile_5_0, MM2S, 5) {
        aie.dma_bd(%arg0 : memref<8192xi32>, 5120, 1024) {bd_id = 10 : i32}
        aie.end
      }
      %t5_1 = aiex.dma_configure_task(%tile_5_1, S2MM, 0) {
        aie.dma_bd(%buf_5 : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t5_0)
      aiex.dma_start_task(%t5_1)
      aiex.dma_await_task(%t5_1)
      %t5_2 = aiex.dma_configure_task(%tile_5_1, MM2S, 0) {
        aie.dma_bd(%buf_5 : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32}
        aie.end
      }
      %t5_3 = aiex.dma_configure_task(%tile_5_0, S2MM, 2) {
        aie.dma_bd(%arg1 : memref<8192xi32>, 5120, 1024) {bd_id = 11 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t5_2)
      aiex.dma_start_task(%t5_3)
      aiex.dma_await_task(%t5_3)

      // Column 6: MM2S ch6, S2MM ch1, BD 12/13
      %t6_0 = aiex.dma_configure_task(%tile_6_0, MM2S, 6) {
        aie.dma_bd(%arg0 : memref<8192xi32>, 6144, 1024) {bd_id = 12 : i32}
        aie.end
      }
      %t6_1 = aiex.dma_configure_task(%tile_6_1, S2MM, 0) {
        aie.dma_bd(%buf_6 : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t6_0)
      aiex.dma_start_task(%t6_1)
      aiex.dma_await_task(%t6_1)
      %t6_2 = aiex.dma_configure_task(%tile_6_1, MM2S, 0) {
        aie.dma_bd(%buf_6 : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32}
        aie.end
      }
      %t6_3 = aiex.dma_configure_task(%tile_6_0, S2MM, 1) {
        aie.dma_bd(%arg1 : memref<8192xi32>, 6144, 1024) {bd_id = 13 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t6_2)
      aiex.dma_start_task(%t6_3)
      aiex.dma_await_task(%t6_3)

      // Column 7: MM2S ch7, S2MM ch0, BD 14/15
      %t7_0 = aiex.dma_configure_task(%tile_7_0, MM2S, 7) {
        aie.dma_bd(%arg0 : memref<8192xi32>, 7168, 1024) {bd_id = 14 : i32}
        aie.end
      }
      %t7_1 = aiex.dma_configure_task(%tile_7_1, S2MM, 0) {
        aie.dma_bd(%buf_7 : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t7_0)
      aiex.dma_start_task(%t7_1)
      aiex.dma_await_task(%t7_1)
      %t7_2 = aiex.dma_configure_task(%tile_7_1, MM2S, 0) {
        aie.dma_bd(%buf_7 : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32}
        aie.end
      }
      %t7_3 = aiex.dma_configure_task(%tile_7_0, S2MM, 0) {
        aie.dma_bd(%arg1 : memref<8192xi32>, 7168, 1024) {bd_id = 15 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t7_2)
      aiex.dma_start_task(%t7_3)
      aiex.dma_await_task(%t7_3)
    }
  }
}

//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Three placed and routed configurations of the same three cores, run in turn
// by @main. Each core copies 64 elements: in @first and @third, the head of its
// weight block; in @second, its part of @first's output. If loading @second
// left @first's connections in place, the outputs of cores (0, 2) and (0, 3)
// would also go down those connections and wait there, and @third's output
// channels would receive them ahead of its own data.

module {
  aie.device(npu3) @first {
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_0_3 = aie.lock(%tile_0_3, 5) {init = 1 : i32}
    %lock_0_3_5 = aie.lock(%tile_0_3, 4) {init = 0 : i32}
    %lock_0_3_6 = aie.lock(%tile_0_3, 3) {init = 1 : i32}
    %lock_0_3_7 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %lock_0_3_8 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %lock_0_3_9 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_0_4 = aie.lock(%tile_0_4, 5) {init = 1 : i32}
    %lock_0_4_10 = aie.lock(%tile_0_4, 4) {init = 0 : i32}
    %lock_0_4_11 = aie.lock(%tile_0_4, 3) {init = 1 : i32}
    %lock_0_4_12 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
    %lock_0_4_13 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
    %lock_0_4_14 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %buf8 = aie.buffer(%tile_0_4) {sym_name = "buf8"} : memref<192xbf16, 2 : i32>
    %buf7 = aie.buffer(%tile_0_4) {sym_name = "buf7"} : memref<12288xbf16, 2 : i32>
    %buf6 = aie.buffer(%tile_0_4) {sym_name = "buf6"} : memref<64xbf16, 2 : i32>
    %buf5 = aie.buffer(%tile_0_3) {sym_name = "buf5"} : memref<192xbf16, 2 : i32>
    %buf4 = aie.buffer(%tile_0_3) {sym_name = "buf4"} : memref<12288xbf16, 2 : i32>
    %buf3 = aie.buffer(%tile_0_3) {sym_name = "buf3"} : memref<64xbf16, 2 : i32>
    %buf2 = aie.buffer(%tile_0_2) {sym_name = "buf2"} : memref<192xbf16, 2 : i32>
    %buf1 = aie.buffer(%tile_0_2) {sym_name = "buf1"} : memref<12288xbf16, 2 : i32>
    %buf0 = aie.buffer(%tile_0_2) {sym_name = "buf0"} : memref<64xbf16, 2 : i32>
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<192xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<110592xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<192xbf16>
    %mem_0_4 = aie.mem(%tile_0_4) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_14, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf6 : memref<64xbf16, 2 : i32> offset = 0 len = 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_13, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_4_11, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf8 : memref<192xbf16, 2 : i32> offset = 0 len = 192) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_12, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf7 : memref<12288xbf16, 2 : i32> offset = 0 len = 12288) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_10, Release, %c1_i32)
      aie.next_bd ^bb6
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c1_i32 = arith.constant 1 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_13, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_4_12, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_4_10, AcquireGreaterEqual, %c1_i32)
      %cp0 = arith.constant 0 : index
      %cp1 = arith.constant 1 : index
      %cp64 = arith.constant 64 : index
      scf.for %i = %cp0 to %cp64 step %cp1 {
        %v = memref.load %buf7[%i] : memref<12288xbf16, 2 : i32>
        memref.store %v, %buf6[%i] : memref<64xbf16, 2 : i32>
      }
      aie.use_lock(%lock_0_4_11, Release, %c1_i32)
      aie.use_lock(%lock_0_4, Release, %c1_i32)
      aie.use_lock(%lock_0_4_14, Release, %c1_i32)
      cf.br ^bb1
    } {stack_size = 2048 : i32}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_9, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf3 : memref<64xbf16, 2 : i32> offset = 0 len = 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_8, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_3_6, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf5 : memref<192xbf16, 2 : i32> offset = 0 len = 192) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_7, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf4 : memref<12288xbf16, 2 : i32> offset = 0 len = 12288) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_5, Release, %c1_i32)
      aie.next_bd ^bb6
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c1_i32 = arith.constant 1 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_8, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_3_7, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_3_5, AcquireGreaterEqual, %c1_i32)
      %cp0 = arith.constant 0 : index
      %cp1 = arith.constant 1 : index
      %cp64 = arith.constant 64 : index
      scf.for %i = %cp0 to %cp64 step %cp1 {
        %v = memref.load %buf4[%i] : memref<12288xbf16, 2 : i32>
        memref.store %v, %buf3[%i] : memref<64xbf16, 2 : i32>
      }
      aie.use_lock(%lock_0_3_6, Release, %c1_i32)
      aie.use_lock(%lock_0_3, Release, %c1_i32)
      aie.use_lock(%lock_0_3_9, Release, %c1_i32)
      cf.br ^bb1
    } {stack_size = 2048 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf0 : memref<64xbf16, 2 : i32> offset = 0 len = 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_3, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf2 : memref<192xbf16, 2 : i32> offset = 0 len = 192) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_2, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf1 : memref<12288xbf16, 2 : i32> offset = 0 len = 12288) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, %c1_i32)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, %c1_i32)
      %cp0 = arith.constant 0 : index
      %cp1 = arith.constant 1 : index
      %cp64 = arith.constant 64 : index
      scf.for %i = %cp0 to %cp64 step %cp1 {
        %v = memref.load %buf1[%i] : memref<12288xbf16, 2 : i32>
        memref.store %v, %buf0[%i] : memref<64xbf16, 2 : i32>
      }
      aie.use_lock(%lock_0_2_1, Release, %c1_i32)
      aie.use_lock(%lock_0_2, Release, %c1_i32)
      aie.use_lock(%lock_0_2_4, Release, %c1_i32)
      cf.br ^bb1
    } {stack_size = 2048 : i32}
    aie.flow(%shim_noc_tile_0_0, DMA : 3, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 3, %tile_0_3, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 3, %tile_0_4, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 1)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_3, DMA : 1)
    aie.flow(%shim_noc_tile_0_0, DMA : 2, %tile_0_4, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %shim_noc_tile_0_0, DMA : 1)
    aie.flow(%tile_0_4, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.shim_dma_allocation @air_channel_2_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_2_1(%shim_noc_tile_0_0, S2MM, 1)
    aie.shim_dma_allocation @air_channel_2_2(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0(%shim_noc_tile_0_0, MM2S, 3)
    aie.shim_dma_allocation @air_channel_1_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_1_1(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_1_2(%shim_noc_tile_0_0, MM2S, 2)
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_0_0, Control32 : 0>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_0_0, Control32 : 1>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%shim_noc_tile_1_0, Control32 : 0>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%shim_noc_tile_1_0, Control32 : 1>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }
    aie.runtime_sequence @first_sequence(%arg0: memref<192xbf16>, %arg1: memref<110592xbf16>, %arg2: memref<192xbf16>, %arg3: memref<192xbf16>, %arg4: memref<192xbf16>, %arg5: memref<192xbf16>) {
      %0 = aiex.dma_configure_task_for @air_channel_2_0 {
        aie.dma_bd(%arg2 : memref<192xbf16> offset = 0 len = 64 sizes = [64] strides = [1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_2_1 {
        aie.dma_bd(%arg2 : memref<192xbf16> offset = 64 len = 64 sizes = [64] strides = [1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_channel_2_2 {
        aie.dma_bd(%arg2 : memref<192xbf16> offset = 128 len = 64 sizes = [64] strides = [1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @air_channel_0 {
        aie.dma_bd(%arg0 : memref<192xbf16> offset = 0 len = 192 sizes = [192] strides = [1])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @air_channel_1_0 {
        aie.dma_bd(%arg1 : memref<110592xbf16> offset = 0 len = 12288 sizes = [12288] strides = [1])
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @air_channel_1_1 {
        aie.dma_bd(%arg1 : memref<110592xbf16> offset = 12288 len = 12288 sizes = [12288] strides = [1])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @air_channel_1_2 {
        aie.dma_bd(%arg1 : memref<110592xbf16> offset = 24576 len = 12288 sizes = [12288] strides = [1])
        aie.end
      }
      aiex.dma_start_task(%6)
      aiex.dma_await_task(%0)
      aiex.dma_await_task(%1)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
    }
  }
  aie.device(npu3) @second {
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_0_3 = aie.lock(%tile_0_3, 5) {init = 1 : i32}
    %lock_0_3_5 = aie.lock(%tile_0_3, 4) {init = 0 : i32}
    %lock_0_3_6 = aie.lock(%tile_0_3, 3) {init = 1 : i32}
    %lock_0_3_7 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %lock_0_3_8 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %lock_0_3_9 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_0_4 = aie.lock(%tile_0_4, 5) {init = 1 : i32}
    %lock_0_4_10 = aie.lock(%tile_0_4, 4) {init = 0 : i32}
    %lock_0_4_11 = aie.lock(%tile_0_4, 3) {init = 1 : i32}
    %lock_0_4_12 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
    %lock_0_4_13 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
    %lock_0_4_14 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %buf17 = aie.buffer(%tile_0_4) {sym_name = "buf17"} : memref<64xbf16, 2 : i32>
    %buf16 = aie.buffer(%tile_0_4) {sym_name = "buf16"} : memref<64xbf16, 2 : i32>
    %buf15 = aie.buffer(%tile_0_4) {sym_name = "buf15"} : memref<64xbf16, 2 : i32>
    %buf14 = aie.buffer(%tile_0_3) {sym_name = "buf14"} : memref<64xbf16, 2 : i32>
    %buf13 = aie.buffer(%tile_0_3) {sym_name = "buf13"} : memref<64xbf16, 2 : i32>
    %buf12 = aie.buffer(%tile_0_3) {sym_name = "buf12"} : memref<64xbf16, 2 : i32>
    %buf11 = aie.buffer(%tile_0_2) {sym_name = "buf11"} : memref<64xbf16, 2 : i32>
    %buf10 = aie.buffer(%tile_0_2) {sym_name = "buf10"} : memref<64xbf16, 2 : i32>
    %buf9 = aie.buffer(%tile_0_2) {sym_name = "buf9"} : memref<64xbf16, 2 : i32>
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<192xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<192xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<192xbf16>
    %mem_0_4 = aie.mem(%tile_0_4) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_14, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf15 : memref<64xbf16, 2 : i32> offset = 0 len = 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_13, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_4_11, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf17 : memref<64xbf16, 2 : i32> offset = 0 len = 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_12, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf16 : memref<64xbf16, 2 : i32> offset = 0 len = 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_10, Release, %c1_i32)
      aie.next_bd ^bb6
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c1_i32 = arith.constant 1 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_13, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_4_12, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_4_10, AcquireGreaterEqual, %c1_i32)
      %cp0 = arith.constant 0 : index
      %cp1 = arith.constant 1 : index
      %cp64 = arith.constant 64 : index
      scf.for %i = %cp0 to %cp64 step %cp1 {
        %v = memref.load %buf17[%i] : memref<64xbf16, 2 : i32>
        memref.store %v, %buf15[%i] : memref<64xbf16, 2 : i32>
      }
      aie.use_lock(%lock_0_4_11, Release, %c1_i32)
      aie.use_lock(%lock_0_4, Release, %c1_i32)
      aie.use_lock(%lock_0_4_14, Release, %c1_i32)
      cf.br ^bb1
    } {stack_size = 2048 : i32}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_9, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf12 : memref<64xbf16, 2 : i32> offset = 0 len = 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_8, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_3_6, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf14 : memref<64xbf16, 2 : i32> offset = 0 len = 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_7, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf13 : memref<64xbf16, 2 : i32> offset = 0 len = 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_5, Release, %c1_i32)
      aie.next_bd ^bb6
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c1_i32 = arith.constant 1 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_8, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_3_7, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_3_5, AcquireGreaterEqual, %c1_i32)
      %cp0 = arith.constant 0 : index
      %cp1 = arith.constant 1 : index
      %cp64 = arith.constant 64 : index
      scf.for %i = %cp0 to %cp64 step %cp1 {
        %v = memref.load %buf14[%i] : memref<64xbf16, 2 : i32>
        memref.store %v, %buf12[%i] : memref<64xbf16, 2 : i32>
      }
      aie.use_lock(%lock_0_3_6, Release, %c1_i32)
      aie.use_lock(%lock_0_3, Release, %c1_i32)
      aie.use_lock(%lock_0_3_9, Release, %c1_i32)
      cf.br ^bb1
    } {stack_size = 2048 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf9 : memref<64xbf16, 2 : i32> offset = 0 len = 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_3, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf11 : memref<64xbf16, 2 : i32> offset = 0 len = 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_2, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf10 : memref<64xbf16, 2 : i32> offset = 0 len = 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, %c1_i32)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, %c1_i32)
      %cp0 = arith.constant 0 : index
      %cp1 = arith.constant 1 : index
      %cp64 = arith.constant 64 : index
      scf.for %i = %cp0 to %cp64 step %cp1 {
        %v = memref.load %buf11[%i] : memref<64xbf16, 2 : i32>
        memref.store %v, %buf9[%i] : memref<64xbf16, 2 : i32>
      }
      aie.use_lock(%lock_0_2_1, Release, %c1_i32)
      aie.use_lock(%lock_0_2, Release, %c1_i32)
      aie.use_lock(%lock_0_2_4, Release, %c1_i32)
      cf.br ^bb1
    } {stack_size = 2048 : i32}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 2, %tile_0_4, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 3, %tile_0_2, DMA : 1)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %tile_0_3, DMA : 1)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %tile_0_4, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %shim_noc_tile_0_0, DMA : 1)
    aie.flow(%tile_0_4, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.shim_dma_allocation @air_channel_5_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_5_1(%shim_noc_tile_0_0, S2MM, 1)
    aie.shim_dma_allocation @air_channel_5_2(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_3_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_3_1(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_3_2(%shim_noc_tile_0_0, MM2S, 2)
    aie.shim_dma_allocation @air_channel_4_0(%shim_noc_tile_0_0, MM2S, 3)
    aie.shim_dma_allocation @air_channel_4_1(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_4_2(%shim_noc_tile_1_0, MM2S, 1)
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_0_0, Control32 : 0>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_0_0, Control32 : 1>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%shim_noc_tile_1_0, Control32 : 0>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%shim_noc_tile_1_0, Control32 : 1>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }
    aie.runtime_sequence @second_sequence(%arg0: memref<192xbf16>, %arg1: memref<110592xbf16>, %arg2: memref<192xbf16>, %arg3: memref<192xbf16>, %arg4: memref<192xbf16>, %arg5: memref<192xbf16>) {
      %0 = aiex.dma_configure_task_for @air_channel_5_0 {
        aie.dma_bd(%arg4 : memref<192xbf16> offset = 0 len = 64 sizes = [64] strides = [1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_5_1 {
        aie.dma_bd(%arg4 : memref<192xbf16> offset = 64 len = 64 sizes = [64] strides = [1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_channel_5_2 {
        aie.dma_bd(%arg4 : memref<192xbf16> offset = 128 len = 64 sizes = [64] strides = [1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @air_channel_3_0 {
        aie.dma_bd(%arg2 : memref<192xbf16> offset = 0 len = 64 sizes = [64] strides = [1])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @air_channel_3_1 {
        aie.dma_bd(%arg2 : memref<192xbf16> offset = 64 len = 64 sizes = [64] strides = [1])
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @air_channel_3_2 {
        aie.dma_bd(%arg2 : memref<192xbf16> offset = 128 len = 64 sizes = [64] strides = [1])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @air_channel_4_0 {
        aie.dma_bd(%arg3 : memref<192xbf16> offset = 0 len = 64 sizes = [64] strides = [1])
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @air_channel_4_1 {
        aie.dma_bd(%arg3 : memref<192xbf16> offset = 64 len = 64 sizes = [64] strides = [1])
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @air_channel_4_2 {
        aie.dma_bd(%arg3 : memref<192xbf16> offset = 128 len = 64 sizes = [64] strides = [1])
        aie.end
      }
      aiex.dma_start_task(%8)
      aiex.dma_await_task(%0)
      aiex.dma_await_task(%1)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%4)
    }
  }
  aie.device(npu3) @third {
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_0_3 = aie.lock(%tile_0_3, 5) {init = 1 : i32}
    %lock_0_3_5 = aie.lock(%tile_0_3, 4) {init = 0 : i32}
    %lock_0_3_6 = aie.lock(%tile_0_3, 3) {init = 1 : i32}
    %lock_0_3_7 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %lock_0_3_8 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %lock_0_3_9 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_0_4 = aie.lock(%tile_0_4, 5) {init = 1 : i32}
    %lock_0_4_10 = aie.lock(%tile_0_4, 4) {init = 0 : i32}
    %lock_0_4_11 = aie.lock(%tile_0_4, 3) {init = 1 : i32}
    %lock_0_4_12 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
    %lock_0_4_13 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
    %lock_0_4_14 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %buf26 = aie.buffer(%tile_0_4) {sym_name = "buf26"} : memref<192xbf16, 2 : i32>
    %buf25 = aie.buffer(%tile_0_4) {sym_name = "buf25"} : memref<12288xbf16, 2 : i32>
    %buf24 = aie.buffer(%tile_0_4) {sym_name = "buf24"} : memref<64xbf16, 2 : i32>
    %buf23 = aie.buffer(%tile_0_3) {sym_name = "buf23"} : memref<192xbf16, 2 : i32>
    %buf22 = aie.buffer(%tile_0_3) {sym_name = "buf22"} : memref<12288xbf16, 2 : i32>
    %buf21 = aie.buffer(%tile_0_3) {sym_name = "buf21"} : memref<64xbf16, 2 : i32>
    %buf20 = aie.buffer(%tile_0_2) {sym_name = "buf20"} : memref<192xbf16, 2 : i32>
    %buf19 = aie.buffer(%tile_0_2) {sym_name = "buf19"} : memref<12288xbf16, 2 : i32>
    %buf18 = aie.buffer(%tile_0_2) {sym_name = "buf18"} : memref<64xbf16, 2 : i32>
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<192xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<110592xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<192xbf16>
    %mem_0_4 = aie.mem(%tile_0_4) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_14, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf24 : memref<64xbf16, 2 : i32> offset = 0 len = 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_13, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_4_11, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf26 : memref<192xbf16, 2 : i32> offset = 0 len = 192) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_12, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf25 : memref<12288xbf16, 2 : i32> offset = 0 len = 12288) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_10, Release, %c1_i32)
      aie.next_bd ^bb6
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c1_i32 = arith.constant 1 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_13, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_4_12, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_4_10, AcquireGreaterEqual, %c1_i32)
      %cp0 = arith.constant 0 : index
      %cp1 = arith.constant 1 : index
      %cp64 = arith.constant 64 : index
      scf.for %i = %cp0 to %cp64 step %cp1 {
        %v = memref.load %buf25[%i] : memref<12288xbf16, 2 : i32>
        memref.store %v, %buf24[%i] : memref<64xbf16, 2 : i32>
      }
      aie.use_lock(%lock_0_4_11, Release, %c1_i32)
      aie.use_lock(%lock_0_4, Release, %c1_i32)
      aie.use_lock(%lock_0_4_14, Release, %c1_i32)
      cf.br ^bb1
    } {stack_size = 2048 : i32}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_9, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf21 : memref<64xbf16, 2 : i32> offset = 0 len = 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_8, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_3_6, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf23 : memref<192xbf16, 2 : i32> offset = 0 len = 192) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_7, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf22 : memref<12288xbf16, 2 : i32> offset = 0 len = 12288) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_5, Release, %c1_i32)
      aie.next_bd ^bb6
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c1_i32 = arith.constant 1 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_8, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_3_7, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_3_5, AcquireGreaterEqual, %c1_i32)
      %cp0 = arith.constant 0 : index
      %cp1 = arith.constant 1 : index
      %cp64 = arith.constant 64 : index
      scf.for %i = %cp0 to %cp64 step %cp1 {
        %v = memref.load %buf22[%i] : memref<12288xbf16, 2 : i32>
        memref.store %v, %buf21[%i] : memref<64xbf16, 2 : i32>
      }
      aie.use_lock(%lock_0_3_6, Release, %c1_i32)
      aie.use_lock(%lock_0_3, Release, %c1_i32)
      aie.use_lock(%lock_0_3_9, Release, %c1_i32)
      cf.br ^bb1
    } {stack_size = 2048 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf18 : memref<64xbf16, 2 : i32> offset = 0 len = 64) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_3, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf20 : memref<192xbf16, 2 : i32> offset = 0 len = 192) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_2, Release, %c1_i32)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buf19 : memref<12288xbf16, 2 : i32> offset = 0 len = 12288) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, %c1_i32)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, %c1_i32)
      %cp0 = arith.constant 0 : index
      %cp1 = arith.constant 1 : index
      %cp64 = arith.constant 64 : index
      scf.for %i = %cp0 to %cp64 step %cp1 {
        %v = memref.load %buf19[%i] : memref<12288xbf16, 2 : i32>
        memref.store %v, %buf18[%i] : memref<64xbf16, 2 : i32>
      }
      aie.use_lock(%lock_0_2_1, Release, %c1_i32)
      aie.use_lock(%lock_0_2, Release, %c1_i32)
      aie.use_lock(%lock_0_2_4, Release, %c1_i32)
      cf.br ^bb1
    } {stack_size = 2048 : i32}
    aie.flow(%shim_noc_tile_0_0, DMA : 3, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 3, %tile_0_3, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 3, %tile_0_4, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 1)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_3, DMA : 1)
    aie.flow(%shim_noc_tile_0_0, DMA : 2, %tile_0_4, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %shim_noc_tile_0_0, DMA : 1)
    aie.flow(%tile_0_4, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.shim_dma_allocation @air_channel_8_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_8_1(%shim_noc_tile_0_0, S2MM, 1)
    aie.shim_dma_allocation @air_channel_8_2(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_6(%shim_noc_tile_0_0, MM2S, 3)
    aie.shim_dma_allocation @air_channel_7_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_channel_7_1(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @air_channel_7_2(%shim_noc_tile_0_0, MM2S, 2)
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_0_0, Control32 : 0>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_0_0, Control32 : 1>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%shim_noc_tile_1_0, Control32 : 0>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%shim_noc_tile_1_0, Control32 : 1>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }
    aie.runtime_sequence @third_sequence(%arg0: memref<192xbf16>, %arg1: memref<110592xbf16>, %arg2: memref<192xbf16>, %arg3: memref<192xbf16>, %arg4: memref<192xbf16>, %arg5: memref<192xbf16>) {
      %0 = aiex.dma_configure_task_for @air_channel_8_0 {
        aie.dma_bd(%arg5 : memref<192xbf16> offset = 0 len = 64 sizes = [64] strides = [1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @air_channel_8_1 {
        aie.dma_bd(%arg5 : memref<192xbf16> offset = 64 len = 64 sizes = [64] strides = [1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @air_channel_8_2 {
        aie.dma_bd(%arg5 : memref<192xbf16> offset = 128 len = 64 sizes = [64] strides = [1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @air_channel_6 {
        aie.dma_bd(%arg4 : memref<192xbf16> offset = 0 len = 192 sizes = [192] strides = [1])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @air_channel_7_0 {
        aie.dma_bd(%arg1 : memref<110592xbf16> offset = 73728 len = 12288 sizes = [12288] strides = [1])
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @air_channel_7_1 {
        aie.dma_bd(%arg1 : memref<110592xbf16> offset = 86016 len = 12288 sizes = [12288] strides = [1])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @air_channel_7_2 {
        aie.dma_bd(%arg1 : memref<110592xbf16> offset = 98304 len = 12288 sizes = [12288] strides = [1])
        aie.end
      }
      aiex.dma_start_task(%6)
      aiex.dma_await_task(%0)
      aiex.dma_await_task(%1)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
    }
  }
  aie.device(npu3) {
    aie.runtime_sequence @sequence(%arg0: memref<192xbf16>, %arg1: memref<110592xbf16>, %arg2: memref<192xbf16>, %arg3: memref<192xbf16>, %arg4: memref<192xbf16>, %arg5: memref<192xbf16>) {
      aiex.configure @first {
        aiex.run @first_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (memref<192xbf16>, memref<110592xbf16>, memref<192xbf16>, memref<192xbf16>, memref<192xbf16>, memref<192xbf16>)
      }
      aiex.configure @second {
        aiex.run @second_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (memref<192xbf16>, memref<110592xbf16>, memref<192xbf16>, memref<192xbf16>, memref<192xbf16>, memref<192xbf16>)
      }
      aiex.configure @third {
        aiex.run @third_sequence(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (memref<192xbf16>, memref<110592xbf16>, memref<192xbf16>, memref<192xbf16>, memref<192xbf16>, memref<192xbf16>)
      }
    }
  }
}

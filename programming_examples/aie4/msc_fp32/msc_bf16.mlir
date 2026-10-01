//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// c - a*b with bf16 inputs widened to f32, contraction permitted on both.
// The accumulator is a splat constant (2^127) so the design takes two inputs
// rather than three, which did not place.
module {
  aie.device(npu3) {
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_0_0, Control32 : 0>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%shim_noc_tile_1_0, Control32 : 0>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }
    aie.packet_flow(2) {
      aie.packet_source<%shim_noc_tile_2_0, Control32 : 0>
      aie.packet_dest<%shim_noc_tile_0_0, Controller32 : 0>
    }
    aie.objectfifo @inA0(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>  
    aie.objectfifo @inA1(%shim_noc_tile_1_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>  
    aie.objectfifo @inA2(%shim_noc_tile_2_0, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>  
    aie.objectfifo @inB0(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>  
    aie.objectfifo @inB1(%shim_noc_tile_1_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>  
    aie.objectfifo @inB2(%shim_noc_tile_2_0, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>  
    aie.objectfifo @outC0(%tile_0_2, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xf32>>  
    aie.objectfifo @outC1(%tile_1_2, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1024xf32>>  
    aie.objectfifo @outC2(%tile_2_2, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<1024xf32>>  
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %0 = aie.objectfifo.acquire @outC0(Produce, 1) : memref<1024xf32>
          %1 = aie.objectfifo.acquire @inA0(Consume, 1) : memref<1024xbf16>
          %2 = aie.objectfifo.acquire @inB0(Consume, 1) : memref<1024xbf16>
          %c0_2 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c32 = arith.constant 32 : index
          %pad = arith.constant 0.000000e+00 : bf16
          %acc = arith.constant dense<1.70141183E+38> : vector<32xf32>
          scf.for %arg2 = %c0_2 to %c1024 step %c32 {
            %ab = vector.transfer_read %1[%arg2], %pad {in_bounds = [true]} : memref<1024xbf16>, vector<32xbf16>
            %bb = vector.transfer_read %2[%arg2], %pad {in_bounds = [true]} : memref<1024xbf16>, vector<32xbf16>
            %3 = arith.extf %ab : vector<32xbf16> to vector<32xf32>
            %4 = arith.extf %bb : vector<32xbf16> to vector<32xf32>
            %m = arith.mulf %3, %4 fastmath<contract> : vector<32xf32>
            %d = arith.subf %acc, %m fastmath<contract> : vector<32xf32>
            vector.transfer_write %d, %0[%arg2] {in_bounds = [true]} : vector<32xf32>, memref<1024xf32>
          }
          aie.objectfifo.release @inA0(Consume, 1)
          aie.objectfifo.release @inB0(Consume, 1)
          aie.objectfifo.release @outC0(Produce, 1)
        }
      }
      aie.end
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %0 = aie.objectfifo.acquire @outC1(Produce, 1) : memref<1024xf32>
          %1 = aie.objectfifo.acquire @inA1(Consume, 1) : memref<1024xbf16>
          %2 = aie.objectfifo.acquire @inB1(Consume, 1) : memref<1024xbf16>
          %c0_2 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c32 = arith.constant 32 : index
          %pad = arith.constant 0.000000e+00 : bf16
          %acc = arith.constant dense<1.70141183E+38> : vector<32xf32>
          scf.for %arg2 = %c0_2 to %c1024 step %c32 {
            %ab = vector.transfer_read %1[%arg2], %pad {in_bounds = [true]} : memref<1024xbf16>, vector<32xbf16>
            %bb = vector.transfer_read %2[%arg2], %pad {in_bounds = [true]} : memref<1024xbf16>, vector<32xbf16>
            %3 = arith.extf %ab : vector<32xbf16> to vector<32xf32>
            %4 = arith.extf %bb : vector<32xbf16> to vector<32xf32>
            %m = arith.mulf %3, %4 fastmath<contract> : vector<32xf32>
            %d = arith.subf %acc, %m fastmath<contract> : vector<32xf32>
            vector.transfer_write %d, %0[%arg2] {in_bounds = [true]} : vector<32xf32>, memref<1024xf32>
          }
          aie.objectfifo.release @inA1(Consume, 1)
          aie.objectfifo.release @inB1(Consume, 1)
          aie.objectfifo.release @outC1(Produce, 1)
        }
      }
      aie.end
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %0 = aie.objectfifo.acquire @outC2(Produce, 1) : memref<1024xf32>
          %1 = aie.objectfifo.acquire @inA2(Consume, 1) : memref<1024xbf16>
          %2 = aie.objectfifo.acquire @inB2(Consume, 1) : memref<1024xbf16>
          %c0_2 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c32 = arith.constant 32 : index
          %pad = arith.constant 0.000000e+00 : bf16
          %acc = arith.constant dense<1.70141183E+38> : vector<32xf32>
          scf.for %arg2 = %c0_2 to %c1024 step %c32 {
            %ab = vector.transfer_read %1[%arg2], %pad {in_bounds = [true]} : memref<1024xbf16>, vector<32xbf16>
            %bb = vector.transfer_read %2[%arg2], %pad {in_bounds = [true]} : memref<1024xbf16>, vector<32xbf16>
            %3 = arith.extf %ab : vector<32xbf16> to vector<32xf32>
            %4 = arith.extf %bb : vector<32xbf16> to vector<32xf32>
            %m = arith.mulf %3, %4 fastmath<contract> : vector<32xf32>
            %d = arith.subf %acc, %m fastmath<contract> : vector<32xf32>
            vector.transfer_write %d, %0[%arg2] {in_bounds = [true]} : vector<32xf32>, memref<1024xf32>
          }
          aie.objectfifo.release @inA2(Consume, 1)
          aie.objectfifo.release @inB2(Consume, 1)
          aie.objectfifo.release @outC2(Produce, 1)
        }
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<49152xbf16>, %arg1: memref<49152xbf16>, %arg2: memref<49152xf32>) {
      %0 = aiex.dma_configure_task_for @inA0 {
        aie.dma_bd(%arg0 : memref<49152xbf16> offset = 0 len = 16384 sizes = [1, 1, 1, 16384] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      %1 = aiex.dma_configure_task_for @inB0 {
        aie.dma_bd(%arg1 : memref<49152xbf16> offset = 0 len = 16384 sizes = [1, 1, 1, 16384] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      %2 = aiex.dma_configure_task_for @outC0 {
        aie.dma_bd(%arg2 : memref<49152xf32> offset = 0 len = 16384 sizes = [1, 1, 1, 16384] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @inA1 {
        aie.dma_bd(%arg0 : memref<49152xbf16> offset = 16384 len = 16384 sizes = [1, 1, 1, 16384] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      %4 = aiex.dma_configure_task_for @inB1 {
        aie.dma_bd(%arg1 : memref<49152xbf16> offset = 16384 len = 16384 sizes = [1, 1, 1, 16384] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      %5 = aiex.dma_configure_task_for @outC1 {
        aie.dma_bd(%arg2 : memref<49152xf32> offset = 16384 len = 16384 sizes = [1, 1, 1, 16384] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%3)
      aiex.dma_start_task(%4)
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @inA2 {
        aie.dma_bd(%arg0 : memref<49152xbf16> offset = 32768 len = 16384 sizes = [1, 1, 1, 16384] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      %7 = aiex.dma_configure_task_for @inB2 {
        aie.dma_bd(%arg1 : memref<49152xbf16> offset = 32768 len = 16384 sizes = [1, 1, 1, 16384] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      %8 = aiex.dma_configure_task_for @outC2 {
        aie.dma_bd(%arg2 : memref<49152xf32> offset = 32768 len = 16384 sizes = [1, 1, 1, 16384] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%6)
      aiex.dma_start_task(%7)
      aiex.dma_start_task(%8)
      aiex.dma_await_task(%0)
      aiex.dma_await_task(%1)
      aiex.dma_await_task(%2)
      aiex.dma_await_task(%3)
      aiex.dma_await_task(%4)
      aiex.dma_await_task(%5)
      aiex.dma_await_task(%6)
      aiex.dma_await_task(%7)
      aiex.dma_await_task(%8)
    }
  }
}

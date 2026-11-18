//===- npu3_shim_bd_layout.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-dma-tasks-to-npu --aie-dma-to-npu --canonicalize %s | FileCheck %s

// AIE4 shim BD packing via the static (npu.writebd) and dynamic
// (npu.blockwrite_values) paths; expected words match aie-codegen.

// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<9xi32> = dense<[0, 0, 512, 50364416, 8388608, 32, 256, 4195328, 0]>
// CHECK-LABEL: @nd
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 98448 : ui32}
// CHECK: aiex.npu.blockwrite_values(%{{.*}} : i32) values %c0_i32, %c0_i32, %c512_i32, %c50364416_i32, %c8388608_i32, %c32_i32, %c256_i32, %c4195328_i32, %c0_i32

// BD 3 is static; BD 4's unfoldable length forces the dynamic encoder.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.runtime_sequence @nd(%arg0 : memref<2048xi32>) {
    %static = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<2048xi32> offset = 0 len = 512 sizes = [2, 4, 8, 16] strides = [1024, 256, 32, 1]) {bd_id = 3 : i32}
      aie.end
    }
    %c256 = arith.constant 256 : i32
    %len = arith.addi %c256, %c256 : i32
    %dynamic = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<2048xi32> offset = 0 len = %len sizes = [2, 4, 8, 16] strides = [1024, 256, 32, 1]) {bd_id = 4 : i32}
      aie.end
    }
  }
}

// -----

// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<9xi32> = dense<[0, 0, 64, 520093696, 4735, -1811939328, 0, 0, 229]>
// CHECK-LABEL: @upper
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 360496 : ui32}
// CHECK: aiex.npu.address_patch(%{{.*}} : i32) {addr = 360500 : ui32, arg_idx = 0 : i32}

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  %lock3 = aie.lock(%shim, 3)
  %lock4 = aie.lock(%shim, 4)
  aie.runtime_sequence @upper(%arg0 : memref<64xi32>) {
    %c1 = arith.constant 1 : i32
    %task = aiex.dma_configure_task(%shim, MM2S, 2) {
      aie.use_lock(%lock3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64) {bd_id = 17 : i32, next_bd_id = 18 : i32, out_of_order_id = 7 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
      aie.use_lock(%lock4, Release, %c1)
      aie.end
    }
  }
}

// -----

// CHECK-LABEL: @runtime_id
// CHECK-DAG: %[[BASE:.*]] = arith.constant 359680 : i32
// CHECK-DAG: %[[STRIDE:.*]] = arith.constant 48 : i32
// CHECK: %[[OFF:.*]] = arith.muli %arg1, %[[STRIDE]] : i32
// CHECK: %[[ADDR:.*]] = arith.addi %[[OFF]], %[[BASE]] : i32
// CHECK: aiex.npu.blockwrite_values(%[[ADDR]] : i32)

// MM2S 2 is in the upper bank: register base 0x58000 + (bd_id - 16) * 0x30.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.runtime_sequence @runtime_id(%arg0 : memref<64xi32>, %id : i32) {
    %task = aiex.dma_configure_task(%shim, MM2S, 2) {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64) bd_id_val %id : i32
      aie.end
    }
  }
}

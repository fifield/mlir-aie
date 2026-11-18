//===- npu3_shim_bd_layout.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// AIE4 shim 9-word BD packing; expected words match aie-codegen's
// XAie_DmaWriteBd for XAIE_DEV_GEN_AIE4.

// RUN: aie-opt --split-input-file --aie-dma-to-npu %s | FileCheck %s
//
// Words: 1 addr low, 2 length, 3 lock acq/burst (512 B: 3)/d1,d2 wraps, 4 d0
// wrap/lock rel, 5 pkt enable/next BD/d1 step, 6 iter current/d2 step,
// 7 AxCache/iter wrap+step, 8 d3 step/out-of-order id/packet id.

// CHECK-LABEL: aie.device(npu3)
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<9xi32> = dense<[0, 0, 256, 50331648, 0, 0, 0, 0, 0]>
// CHECK-LABEL: @linear
// Shim BD 0 at 0x18000; the address patch targets its word 1.
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 98304 : ui32}
// CHECK: aiex.npu.address_patch(%{{.*}} : i32) {addr = 98308 : ui32, arg_idx = 0 : i32}

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @in0(%shim, MM2S, 0)
    aie.runtime_sequence @linear(%arg0 : memref<256xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, metadata = @in0} : memref<256xi32>
    }
  }
}

// -----

// CHECK-LABEL: aie.device(npu3)
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<9xi32> = dense<[0, 0, 512, 50364416, 8388608, 32, 256, 4195328, 0]>
// CHECK-LABEL: @nd
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 98448 : ui32}
// CHECK: aiex.npu.address_patch(%{{.*}} : i32) {addr = 98452 : ui32, arg_idx = 0 : i32}
// Repeat 1 (two iterations), BD 3.
// CHECK: %[[CMD:.*]] = arith.constant 65539 : i32
// CHECK: %[[Q:.*]] = arith.constant 99668 : i32
// CHECK: aiex.npu.write32(%[[Q]], %[[CMD]])

// AIE4 BD steps are the strides themselves, not stride - 1.
module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @in0(%shim, MM2S, 0)
    aie.runtime_sequence @nd(%arg0 : memref<2048xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][2, 4, 8, 16][1024, 256, 32, 1]) {id = 3 : i64, metadata = @in0} : memref<2048xi32>
    }
  }
}

// -----

// CHECK-LABEL: aie.device(npu3)
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<9xi32> = dense<[0, 0, 64, 50331648, 0, -2147483648, 0, 0, 5]>
// CHECK-LABEL: @upper
// Upper-bank BD 1 at 0x58030; the task queue names it bank-locally.
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 360496 : ui32}
// CHECK: aiex.npu.address_patch(%{{.*}} : i32) {addr = 360500 : ui32, arg_idx = 0 : i32}
// CHECK: %[[CMD:.*]] = arith.constant 1 : i32
// CHECK: %[[Q:.*]] = arith.constant 361812 : i32
// CHECK: aiex.npu.write32(%[[Q]], %[[CMD]])

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @in2(%shim, MM2S, 2)
    aie.runtime_sequence @upper(%arg0 : memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 5>) {id = 17 : i64, metadata = @in2} : memref<64xi32>
    }
  }
}

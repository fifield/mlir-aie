// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --aie-dma-to-npu --canonicalize %s | FileCheck %s
// Runtime offsets must select the same nine-register MDS layout as static BDs.
// CHECK-LABEL: @offset_d3
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[LEN:.*]] = arith.constant 48 : i32
// CHECK-DAG: %[[W3:.*]] = arith.constant 50343938 : i32
// CHECK-DAG: %[[W4:.*]] = arith.constant 2097152 : i32
// CHECK-DAG: %[[W5:.*]] = arith.constant 4 : i32
// CHECK-DAG: %[[W6:.*]] = arith.constant 12 : i32
// CHECK-DAG: %[[W8:.*]] = arith.constant 30720 : i32
// CHECK: aiex.npu.blockwrite_values({{.*}}) values %[[ZERO]], %[[ZERO]], %[[LEN]], %[[W3]], %[[W4]], %[[W5]], %[[W6]], %[[ZERO]], %[[W8]] : i32
module {
 aie.device(npu3) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @input (%tile, MM2S, 0)
  aie.runtime_sequence @offset_d3(%in: memref<128xi32>, %off: i64) {
   aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0, %off][1, 2, 2, 3, 4][0, 30, 12, 4, 1]) {metadata = @input, id = 0 : i64} : memref<128xi32>
  }
 }
}

// -----
// Four entries continue to mean iteration,D2,D1,D0, including on MDS.
// CHECK-LABEL: @legacy_repeat
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[LEN:.*]] = arith.constant 24 : i32
// CHECK-DAG: %[[W3:.*]] = arith.constant 50343938 : i32
// CHECK-DAG: %[[W4:.*]] = arith.constant 2097152 : i32
// CHECK-DAG: %[[W5:.*]] = arith.constant 4 : i32
// CHECK-DAG: %[[W6:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[ITER:.*]] = arith.constant 4194404 : i32
// CHECK: aiex.npu.blockwrite_values({{.*}}) values %[[ZERO]], %[[ZERO]], %[[LEN]], %[[W3]], %[[W4]], %[[W5]], %[[W6]], %[[ITER]], %[[ZERO]] : i32
module {
 aie.device(npu3) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @input (%tile, MM2S, 0)
  aie.runtime_sequence @legacy_repeat(%in: memref<256xi32>, %off: i64) {
   aiex.npu.dma_memcpy_nd(%in[0, 0, 0, %off][2, 2, 3, 4][100, 16, 4, 1]) {metadata = @input, id = 0 : i64} : memref<256xi32>
  }
 }
}

// -----
// A runtime offset preserves the contiguous transfer's wide length exemption.
// CHECK-LABEL: @linear
// CHECK: arith.constant 8192 : i32
// CHECK: aiex.npu.blockwrite_values
module {
 aie.device(npu3) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @input (%tile, MM2S, 0)
  aie.runtime_sequence @linear(%in: memref<16384xi32>, %off: i64) {
   aiex.npu.dma_memcpy_nd(%in[0, 0, 0, %off][1, 1, 1, 8192][0, 0, 0, 1]) {metadata = @input, id = 0 : i64} : memref<16384xi32>
  }
 }
}

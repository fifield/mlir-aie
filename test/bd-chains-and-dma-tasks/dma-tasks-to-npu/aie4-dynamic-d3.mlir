// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --aie-normalize-dma-bd-dims --aie-substitute-shim-dma-allocations --aie-dma-tasks-to-npu --canonicalize %s | FileCheck %s

// Runtime size independently selects the shared MDS encoder.
// CHECK-LABEL: @size
// CHECK: aiex.npu.assert_bd_field({{.*}}) {max = 0 : i32}
// CHECK: aiex.npu.assert_bd_field({{.*}}) {max = 4095 : i32}
// CHECK: aiex.npu.blockwrite_values({{.*}}) values {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}} : i32
module {
 aie.device(npu3) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @input (%tile, MM2S, 0)
  aie.runtime_sequence @size(%in: memref<128xi32>, %n: i64, %id: i32) {
   %t = aiex.dma_configure_task_for @input {
    aie.dma_bd(%in : memref<128xi32> offset = 0 len = 48 sizes = [1, 2, 2, %n, 4] strides = [0, 30, 12, 4, 1]) {bd_id = 0 : i32}
    aie.end
   }
   aiex.dma_start_task(%t)
  }
 }
}

// -----
// Runtime stride independently selects the shared MDS encoder.
// CHECK-LABEL: @stride
// CHECK: aiex.npu.assert_bd_field({{.*}}) {max = 4194303 : i32}
// CHECK: aiex.npu.blockwrite_values({{.*}}) values {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}} : i32
module {
 aie.device(npu3) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @input (%tile, MM2S, 0)
  aie.runtime_sequence @stride(%in: memref<128xi32>, %n: i64, %id: i32) {
   %t = aiex.dma_configure_task_for @input {
    aie.dma_bd(%in : memref<128xi32> offset = 0 len = 48 sizes = [1, 2, 2, 3, 4] strides = [0, %n, 12, 4, 1]) {bd_id = 0 : i32}
    aie.end
   }
   aiex.dma_start_task(%t)
  }
 }
}

// -----
// Runtime length independently selects the shared MDS encoder.
// CHECK-LABEL: @length
// CHECK: aiex.npu.assert_bd_field({{.*}}) {max = 0 : i32}
// CHECK: aiex.npu.blockwrite_values({{.*}}) values {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}} : i32
module {
 aie.device(npu3) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @input (%tile, MM2S, 0)
  aie.runtime_sequence @length(%in: memref<128xi32>, %n: i32, %id: i32) {
   %t = aiex.dma_configure_task_for @input {
    aie.dma_bd(%in : memref<128xi32> offset = 0 len = %n sizes = [1, 2, 2, 3, 4] strides = [0, 30, 12, 4, 1]) {bd_id = 0 : i32}
    aie.end
   }
   aiex.dma_start_task(%t)
  }
 }
}

// -----
// Runtime bd_id independently selects the shared MDS encoder.
// CHECK-LABEL: @bd_id
// CHECK: aiex.npu.blockwrite_values({{.*}}) values {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}} : i32
module {
 aie.device(npu3) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @input (%tile, MM2S, 0)
  aie.runtime_sequence @bd_id(%in: memref<128xi32>, %n: i64, %id: i32) {
   %t = aiex.dma_configure_task_for @input {
    aie.dma_bd(%in : memref<128xi32> offset = 0 len = 48 sizes = [1, 2, 2, 3, 4] strides = [0, 30, 12, 4, 1]) bd_id_val %id : i32
    aie.end
   }
   aiex.dma_start_task(%t)
  }
 }
}

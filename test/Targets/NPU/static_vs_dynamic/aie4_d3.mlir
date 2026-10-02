// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REQUIRES: peano
// The static binary emitter and dynamic host builder must program all nine
// MDS BD registers identically for each independent runtime operand.
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: aie-opt --aie-normalize-dma-bd-dims --aie-substitute-shim-dma-allocations --aie-dma-tasks-to-npu --aie-dma-to-npu %s -o %t.d/lowered.mlir
// RUN: aie-translate --aie-npu-to-cpp %t.d/lowered.mlir > %t.d/gen.h
// RUN: %host_clang -std=c++17 -I%S/../../../../include -DGEN_HDR='"%t.d/gen.h"' %S/Inputs/aie4_d3_invalid.cpp %host_link_flags -o %t.d/invalid.exe
// RUN: %t.d/invalid.exe
// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false -aie-sequence-name=offset_static %t.d/lowered.mlir > %t.d/offset.hex
// RUN: %host_clang -std=c++17 -DDYN_STRUCTURAL -I%S/../../../../include -DGEN_HDR='"%t.d/gen.h"' -DSTATIC_FN=generate_txn_main_offset_static -DDYN_FN=generate_txn_main_offset_dynamic -DARGVAL=3 %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/offset.exe
// RUN: %t.d/offset.exe %t.d/offset.hex
// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false -aie-sequence-name=size_static %t.d/lowered.mlir > %t.d/size.hex
// RUN: %host_clang -std=c++17 -DDYN_STRUCTURAL -I%S/../../../../include -DGEN_HDR='"%t.d/gen.h"' -DSTATIC_FN=generate_txn_main_size_static -DDYN_FN=generate_txn_main_size_dynamic -DARGVAL=3 %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/size.exe
// RUN: %t.d/size.exe %t.d/size.hex
// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false -aie-sequence-name=stride_static %t.d/lowered.mlir > %t.d/stride.hex
// RUN: %host_clang -std=c++17 -DDYN_STRUCTURAL -I%S/../../../../include -DGEN_HDR='"%t.d/gen.h"' -DSTATIC_FN=generate_txn_main_stride_static -DDYN_FN=generate_txn_main_stride_dynamic -DARGVAL=4 %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/stride.exe
// RUN: %t.d/stride.exe %t.d/stride.hex
// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false -aie-sequence-name=length_static %t.d/lowered.mlir > %t.d/length.hex
// RUN: %host_clang -std=c++17 -DDYN_STRUCTURAL -I%S/../../../../include -DGEN_HDR='"%t.d/gen.h"' -DSTATIC_FN=generate_txn_main_length_static -DDYN_FN=generate_txn_main_length_dynamic -DARGVAL=48 %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/length.exe
// RUN: %t.d/length.exe %t.d/length.hex
// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false -aie-sequence-name=id_static %t.d/lowered.mlir > %t.d/id.hex
// RUN: %host_clang -std=c++17 -DDYN_STRUCTURAL -I%S/../../../../include -DGEN_HDR='"%t.d/gen.h"' -DSTATIC_FN=generate_txn_main_id_static -DDYN_FN=generate_txn_main_id_dynamic -DARGVAL=1 %S/Inputs/compare_main.cpp %host_link_flags -o %t.d/id.exe
// RUN: %t.d/id.exe %t.d/id.hex

module {
 aie.device(npu3) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @input (%tile, MM2S, 0)
  aie.runtime_sequence @offset_static(%in: memref<128xi32>) {
   aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0, 3][1, 2, 2, 3, 4][0, 30, 12, 4, 1]) {metadata = @input, id = 0 : i64} : memref<128xi32>
  }
  aie.runtime_sequence @offset_dynamic(%in: memref<128xi32>, %n: i64) {
   aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0, %n][1, 2, 2, 3, 4][0, 30, 12, 4, 1]) {metadata = @input, id = 0 : i64} : memref<128xi32>
  }
  aie.runtime_sequence @size_static(%in: memref<128xi32>) {
   %t = aiex.dma_configure_task_for @input {
    aie.dma_bd(%in : memref<128xi32> offset = 0 len = 48 sizes = [1, 2, 2, 3, 4] strides = [0, 30, 12, 4, 1]) {bd_id = 0 : i32}
    aie.end
   }
   aiex.dma_start_task(%t)
  }
  aie.runtime_sequence @size_dynamic(%in: memref<128xi32>, %n: i64) {
   %t = aiex.dma_configure_task_for @input {
    aie.dma_bd(%in : memref<128xi32> offset = 0 len = 48 sizes = [1, 2, 2, %n, 4] strides = [0, 30, 12, 4, 1]) {bd_id = 0 : i32}
    aie.end
   }
   aiex.dma_start_task(%t)
  }
  aie.runtime_sequence @stride_static(%in: memref<128xi32>) {
   %t = aiex.dma_configure_task_for @input {
    aie.dma_bd(%in : memref<128xi32> offset = 0 len = 48 sizes = [1, 2, 2, 3, 4] strides = [0, 30, 12, 4, 1]) {bd_id = 0 : i32}
    aie.end
   }
   aiex.dma_start_task(%t)
  }
  aie.runtime_sequence @stride_dynamic(%in: memref<128xi32>, %n: i64) {
   %t = aiex.dma_configure_task_for @input {
    aie.dma_bd(%in : memref<128xi32> offset = 0 len = 48 sizes = [1, 2, 2, 3, 4] strides = [0, 30, 12, %n, 1]) {bd_id = 0 : i32}
    aie.end
   }
   aiex.dma_start_task(%t)
  }
  aie.runtime_sequence @length_static(%in: memref<128xi32>) {
   %t = aiex.dma_configure_task_for @input {
    aie.dma_bd(%in : memref<128xi32> offset = 0 len = 48 sizes = [1, 2, 2, 3, 4] strides = [0, 30, 12, 4, 1]) {bd_id = 0 : i32}
    aie.end
   }
   aiex.dma_start_task(%t)
  }
  aie.runtime_sequence @length_dynamic(%in: memref<128xi32>, %n: i32) {
   %t = aiex.dma_configure_task_for @input {
    aie.dma_bd(%in : memref<128xi32> offset = 0 len = %n sizes = [1, 2, 2, 3, 4] strides = [0, 30, 12, 4, 1]) {bd_id = 0 : i32}
    aie.end
   }
   aiex.dma_start_task(%t)
  }
  aie.runtime_sequence @wide_length(%in: memref<128xi128>, %n: i32) {
   %t = aiex.dma_configure_task_for @input {
    aie.dma_bd(%in : memref<128xi128> offset = 0 len = %n) {bd_id = 0 : i32}
    aie.end
   }
   aiex.dma_start_task(%t)
  }
  aie.runtime_sequence @wide_stride(%in: memref<128xi128>, %n: i64) {
   %t = aiex.dma_configure_task_for @input {
    aie.dma_bd(%in : memref<128xi128> offset = 0 len = 48 sizes = [1, 2, 2, 3, 4] strides = [0, 30, 12, %n, 1]) {bd_id = 0 : i32}
    aie.end
   }
   aiex.dma_start_task(%t)
  }
  aie.runtime_sequence @id_static(%in: memref<128xi32>) {
   %t = aiex.dma_configure_task_for @input {
    aie.dma_bd(%in : memref<128xi32> offset = 0 len = 48 sizes = [1, 2, 2, 3, 4] strides = [0, 30, 12, 4, 1]) {bd_id = 1 : i32}
    aie.end
   }
   aiex.dma_start_task(%t)
  }
  aie.runtime_sequence @id_dynamic(%in: memref<128xi32>, %id: i32) {
   %t = aiex.dma_configure_task_for @input {
    aie.dma_bd(%in : memref<128xi32> offset = 0 len = 48 sizes = [1, 2, 2, 3, 4] strides = [0, 30, 12, 4, 1]) bd_id_val %id : i32
    aie.end
   }
   aiex.dma_start_task(%t)
  }
 }
}

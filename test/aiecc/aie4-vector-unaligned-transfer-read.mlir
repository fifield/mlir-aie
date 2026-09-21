// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compile an unaligned read with AIE4 vectorization enabled. Its slice must
// survive AIEVec legalization and select lanes 3..18 from the widened load.
//
// REQUIRES: peano
// RUN: %aiecc -j 4 --enable-aie4-vectorization --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s
// RUN: FileCheck %s --input-file=%t.prj/opted_main_core_0_2.ll
// RUN: llvm-objdump -dr %t.out/elfs_main_core_0_2/elfs_main_core_0_2.elf > %t.dis

// CHECK-LABEL: define void @core_0_2()
// CHECK: %[[LOAD:[0-9]+]] = load <32 x i32>, ptr @src
// CHECK: %[[SLICE:[0-9]+]] = shufflevector <32 x i32> %[[LOAD]], <32 x i32> poison, <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
// CHECK: store <16 x i32> %[[SLICE]], ptr @dst

module {
  aie.device(npu3) {
    %tile = aie.tile(0, 2)
    %src = aie.buffer(%tile) {sym_name = "src"} : memref<64xi32>
    %dst = aie.buffer(%tile) {sym_name = "dst"} : memref<16xi32>
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c3 = arith.constant 3 : index
      %pad = arith.constant 0 : i32
      %v = vector.transfer_read %src[%c3], %pad {in_bounds = [true]} : memref<64xi32>, vector<16xi32>
      vector.transfer_write %v, %dst[%c0] {in_bounds = [true]} : vector<16xi32>, memref<16xi32>
      aie.end
    }
  }
}

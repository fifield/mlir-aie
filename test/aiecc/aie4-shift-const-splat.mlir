// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shifting every lane left by 3 lowers through vector addition on AIE4.
//
// REQUIRES: peano
// RUN: %aiecc -j 4 --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s
// RUN: llvm-objdump -dr %t.out/elfs_main_core_0_2/elfs_main_core_0_2.elf | FileCheck %s

// CHECK: vadd.32

module {
  aie.device(npu3) {
    %tile_0_2 = aie.tile(0, 2)

    %buf_a = aie.buffer(%tile_0_2) {sym_name = "buf_a"} : memref<16xi32>
    %buf_c = aie.buffer(%tile_0_2) {sym_name = "buf_c"} : memref<16xi32>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32

      %a = vector.transfer_read %buf_a[%c0], %c0_i32 {in_bounds = [true]} : memref<16xi32>, vector<16xi32>
      %s = arith.constant dense<3> : vector<16xi32>
      %r = arith.shli %a, %s : vector<16xi32>
      vector.transfer_write %r, %buf_c[%c0] {in_bounds = [true]} : vector<16xi32>, memref<16xi32>

      aie.end
    }
  }
}

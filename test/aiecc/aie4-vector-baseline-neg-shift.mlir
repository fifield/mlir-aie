//===- aie4-vector-baseline-neg-shift.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Per-lane arith.shli on vector<16xi32> fails G_SHL legalization on AIE4.
//
// REQUIRES: peano
//
// RUN: not %aiecc -j 4 --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s 2>&1 | FileCheck %s

// CHECK: LLVM ERROR: unable to legalize instruction:{{.*}}G_SHL

module {
  aie.device(npu3) {
    %tile_0_2 = aie.tile(0, 2)

    %buf_a = aie.buffer(%tile_0_2) {sym_name = "buf_a"} : memref<16xi32>
    %buf_s = aie.buffer(%tile_0_2) {sym_name = "buf_s"} : memref<16xi32>
    %buf_c = aie.buffer(%tile_0_2) {sym_name = "buf_c"} : memref<16xi32>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32

      %a = vector.transfer_read %buf_a[%c0], %c0_i32 {in_bounds = [true]} : memref<16xi32>, vector<16xi32>
      %s = vector.transfer_read %buf_s[%c0], %c0_i32 {in_bounds = [true]} : memref<16xi32>, vector<16xi32>
      %r = arith.shli %a, %s : vector<16xi32>
      vector.transfer_write %r, %buf_c[%c0] {in_bounds = [true]} : vector<16xi32>, memref<16xi32>

      aie.end
    }
  }
}

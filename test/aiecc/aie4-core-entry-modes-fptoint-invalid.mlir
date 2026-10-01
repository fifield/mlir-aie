// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A core that converts float to int (crRnd = 2) and also rounds bf16
// (crRnd = 12) must fail the normal aiecc pipeline.
//
// REQUIRES: peano
// RUN: not %aiecc --no-unified --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s 2>&1 | FileCheck %s
// RUN: not %aiecc --unified --get-core-elfs --tmpdir %t.uprj --output-dir %t.uout %s 2>&1 | FileCheck %s
// CHECK: error: 'aie.core' op mixes a float -> int conversion, which needs crRnd = 2
// CHECK: error: failed to legalize operation 'aie.core'

module {
  aie.device(npu3) {
    %tile = aie.tile(0, 2)
    %input = aie.buffer(%tile) {sym_name = "input"} : memref<32xf32>
    %ints = aie.buffer(%tile) {sym_name = "ints"} : memref<32xi32>
    %half = aie.buffer(%tile) {sym_name = "half"} : memref<32xbf16>
    aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %v = vector.load %input[%c0] : memref<32xf32>, vector<32xf32>
      %i = aievec.aie4.conv_f32_i32 %v : vector<32xf32>, vector<32xi32>
      vector.store %i, %ints[%c0] : memref<32xi32>, vector<32xi32>
      %a = memref.load %half[%c0] : memref<32xbf16>
      %sum = arith.addf %a, %a : bf16
      memref.store %sum, %half[%c0] : memref<32xbf16>
      aie.end
    }
  }
}

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Conflicting rounding requirements must fail the normal aiecc pipeline.
// A failed outlining pattern must not leave the core for device removal and
// let compilation continue without its body.
//
// REQUIRES: peano
// RUN: not %aiecc --no-unified --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s 2>&1 | FileCheck %s
// RUN: not %aiecc --unified --get-core-elfs --tmpdir %t.uprj --output-dir %t.uout %s 2>&1 | FileCheck %s
// CHECK: error: 'aie.core' op mixes ops that need round-to-nearest-even
// CHECK: error: failed to legalize operation 'aie.core'

module {
  aie.device(npu3) {
    %tile = aie.tile(0, 2)
    %input = aie.buffer(%tile) {sym_name = "input"} : memref<32xf32>
    %output = aie.buffer(%tile) {sym_name = "output"} : memref<32xbf16>
    aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %v = vector.load %input[%c0] : memref<32xf32>, vector<32xf32>
      %converted = aievec.aie4.conv_f32_bf16 %v : vector<32xf32>, vector<32xbf16>
      vector.store %converted, %output[%c0] : memref<32xbf16>, vector<32xbf16>
      %a = memref.load %output[%c0] : memref<32xbf16>
      %sum = arith.addf %a, %a : bf16
      memref.store %sum, %output[%c0] : memref<32xbf16>
      aie.end
    }
  }
}

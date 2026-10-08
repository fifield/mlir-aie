// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// aiecc rejects a matmul next to a flat stage in one core.
// REQUIRES: peano
// RUN: not %aiecc --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s 2>&1 | FileCheck %s
// CHECK: error: 'aievec.aie4.matmul' must not share an execution scope with flat staging operations
// CHECK: note: flat staging operation here

module {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %buf = aie.buffer(%t) {sym_name = "out"} : memref<128xf32>
    %core = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %s = arith.constant 0 : i32
      %chunk = arith.constant dense<0.0> : vector<64xbf16>
      %a = arith.constant dense<1.0> : vector<2x32xbf16>
      %b = arith.constant dense<1.0> : vector<32x64xbf16>
      %c = arith.constant dense<0.0> : vector<2x64xf32>
      aievec.aie4.stage bf16 %chunk, %s : vector<64xbf16>, i32
      %r = aievec.aie4.matmul %a, %b, %c : vector<2x32xbf16>, vector<32x64xbf16> into vector<2x64xf32>
      %r0 = vector.extract %r[0] : vector<64xf32> from vector<2x64xf32>
      vector.store %r0, %buf[%c0] : memref<128xf32>, vector<64xf32>
      aie.end
    }
  }
}

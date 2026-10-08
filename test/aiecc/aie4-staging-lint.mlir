// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// aiecc rejects unprimed commits by default.
//
// REQUIRES: peano

// RUN: not %aiecc --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s 2>&1 | FileCheck %s --check-prefix=E1

// E1: error: 'aievec.aie4.commit' reads staging slot 0, which may be unwritten or clobbered on some path

module {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %buf = aie.buffer(%t) {sym_name = "out"} : memref<64xi32>
    %core = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %a = arith.constant dense<1> : vector<64xi8>
      %acc = arith.constant dense<0> : vector<64xi32>
      aievec.aie4.commit int8 {rhs_signed = true}
      %r = aievec.aie4.mmac_active int8 %a, %acc {lhs_signed = true, rhs_signed = true} : vector<64xi8>, vector<64xi32> -> vector<64xi32>
      vector.store %r, %buf[%c0] : memref<64xi32>, vector<64xi32>
      aie.end
    }
  }
}

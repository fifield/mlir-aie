// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REQUIRES: peano
// RUN: %aiecc --enable-aie4-vectorization --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s
// RUN: FileCheck %s --input-file=%t.prj/peano-compat_main_core_0_2.ll
// CHECK-DAG: fadd float
// CHECK-DAG: fsub float
// CHECK-DAG: call <32 x float> @llvm.aie4.I1024.I1024.ACC1024.fp.mul.ewise.conf
module {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<32xf32>
    %b = aie.buffer(%t) {sym_name = "b"} : memref<32xf32>
    %sum = aie.buffer(%t) {sym_name = "sum"} : memref<1xf32>
    %diff = aie.buffer(%t) {sym_name = "diff"} : memref<1xf32>
    %prod = aie.buffer(%t) {sym_name = "prod"} : memref<32xf32>
    %core = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %sa = memref.load %a[%c0] : memref<32xf32>
      %sb = memref.load %b[%c0] : memref<32xf32>
      %s = arith.addf %sa, %sb : f32
      %d = arith.subf %sa, %sb : f32
      memref.store %s, %sum[%c0] : memref<1xf32>
      memref.store %d, %diff[%c0] : memref<1xf32>
      %va = vector.load %a[%c0] : memref<32xf32>, vector<32xf32>
      %vb = vector.load %b[%c0] : memref<32xf32>, vector<32xf32>
      %m = arith.mulf %va, %vb : vector<32xf32>
      vector.store %m, %prod[%c0] : memref<32xf32>, vector<32xf32>
      aie.end
    }
  }
}

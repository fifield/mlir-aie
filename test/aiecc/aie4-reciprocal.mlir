// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REQUIRES: peano
// RUN: %aiecc --enable-aie4-vectorization --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.prj/opted_main_core_0_2.ll
// RUN: llvm-objdump -dr %t.out/elfs_main_core_0_2/elfs_main_core_0_2.elf > %t.dis
// RUN: FileCheck %s --check-prefix=ASM --input-file=%t.dis

// afn 1.0 / x compiles to the INV instruction; a strict 1.0 / x stays an
// exact fdiv, which Peano links as the __divsf3 soft-float routine.

// IR-LABEL: define void @core_0_2()
// IR-DAG: call float @llvm.aie4.inv(float
// IR-DAG: fdiv float 1.000000e+00,
// IR-NOT: fdiv afn

// ASM-DAG: {{[[:space:]]}}inv{{[[:space:]]+}}r{{[0-9]+}}, r{{[0-9]+}}
// ASM-DAG: <__divsf3>:
// ASM-NOT: R_AIE{{.*}}llvm.aie4

module {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %x = aie.buffer(%t) {sym_name = "x"} : memref<1xf32>
    %approx = aie.buffer(%t) {sym_name = "approx"} : memref<1xf32>
    %exact = aie.buffer(%t) {sym_name = "exact"} : memref<1xf32>
    %core = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %one = arith.constant 1.0 : f32
      %v = memref.load %x[%c0] : memref<1xf32>
      %a = arith.divf %one, %v fastmath<afn> : f32
      memref.store %a, %approx[%c0] : memref<1xf32>
      %e = arith.divf %one, %v : f32
      memref.store %e, %exact[%c0] : memref<1xf32>
      aie.end
    }
  }
}

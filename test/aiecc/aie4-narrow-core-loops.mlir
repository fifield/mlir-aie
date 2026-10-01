// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REQUIRES: peano
// RUN: %aiecc --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t.prj/opted_main_core_0_2.ll
// RUN: llvm-objdump -d %t.out/elfs_main_core_0_2/elfs_main_core_0_2.elf | FileCheck %s

// The sentinel loop stays i64; the bounded vector loop becomes i32.
// LLVM-LABEL: define void @core_0_2()
// LLVM:       phi i64
// LLVM:       phi i32
// LLVM:       add <16 x i32>
// LLVM:       icmp {{.*}} i32
// LLVM:       icmp eq i64

// CHECK-LABEL: <main>:
// CHECK:       movxm{{[[:space:]]+}}ls, #0x[[LS:[0-9a-f]+]]
// CHECK:       movxm{{[[:space:]]+}}le, #0x[[LE:[0-9a-f]+]]
// CHECK:       add.nc{{[[:space:]]+}}lc,
// CHECK:       {{^0*}}[[LS]] <.LBB
// CHECK:       vadd.32
// CHECK:       {{^0*}}[[LE]] <.L_LEnd

module {
  aie.device(npu3) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a"} : memref<1024xi32>
    %b = aie.buffer(%t) {sym_name = "b"} : memref<1024xi32>
    %c = aie.buffer(%t) {sym_name = "c"} : memref<1024xi32>
    %lin = aie.lock(%t, 0) {init = 0 : i32, sym_name = "lin"}
    %lout = aie.lock(%t, 1) {init = 1 : i32, sym_name = "lout"}
    %core = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c1024 = arith.constant 1024 : index
      %cmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : i32
      scf.for %it = %c0 to %cmax step %c1 {
        aie.use_lock(%lin, AcquireGreaterEqual, %one)
        aie.use_lock(%lout, AcquireGreaterEqual, %one)
        scf.for %i = %c0 to %c1024 step %c16 {
          %va = vector.load %a[%i] : memref<1024xi32>, vector<16xi32>
          %vb = vector.load %b[%i] : memref<1024xi32>, vector<16xi32>
          %vc = arith.addi %va, %vb : vector<16xi32>
          vector.store %vc, %c[%i] : memref<1024xi32>, vector<16xi32>
        }
        aie.use_lock(%lout, Release, %one)
        aie.use_lock(%lin, Release, %one)
      }
      aie.end
    }
  }
}

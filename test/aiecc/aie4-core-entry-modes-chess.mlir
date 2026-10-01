// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The D19 core-entry writes also compile and link on the Chess path. With
// --xchesscc, Chess compiles the core's LLVM IR and renames
// llvm.aie4.set.ctrl.reg to llvm___aie4___set___ctrl___reg, which the AIE4
// chess_intrinsic_wrapper must define. Without it, every AIE4 --xchesscc
// design failed to link with `undefined symbol
// "llvm___aie4___set___ctrl___reg"`.
//
// REQUIRES: chess, peano
// RUN: rm -rf %t.prj %t.out
// RUN: %aiecc --xchesscc --xbridge --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s
// RUN: %PEANO_INSTALL_DIR/bin/llvm-objdump -d --no-show-raw-insn %t.out/elfs_main_core_0_2/elfs_main_core_0_2.elf | FileCheck %s

// The core passes (24, 12) for crVecRnd and (46, 12) for crRnd.
// CHECK-LABEL: <core_0_2>:
// CHECK-DAG:   mov r0, #0x18
// CHECK-DAG:   mov r0, #0x2e
// CHECK-DAG:   mov r1, #0xc
// CHECK-DAG:   jl
// CHECK-LABEL: <llvm___aie4___set___ctrl___reg>:
// CHECK:       mov crf2bconvsat, r1

module {
  aie.device(npu3) {
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) {sym_name = "buf"} : memref<16xi32>
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %v = arith.constant 7 : i32
      memref.store %v, %buf[%c0] : memref<16xi32>
      aie.end
    }
  }
}

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// An explicit-mode aievec.aie4.srs in a loop: the core writes crRnd and
// crSat once, at entry, and Peano fuses the SRS into the store
// (vst.srs.4x). crsrsmode, which Peano writes itself, matches neither
// count pattern.
//
// REQUIRES: peano
// RUN: %aiecc --get-core-elfs --tmpdir %t.prj --output-dir %t.out %s
// RUN: llvm-objdump -dr --no-show-raw-insn %t.out/elfs_main_core_0_2/elfs_main_core_0_2.elf > %t.s
// RUN: FileCheck %s --input-file=%t.s
// RUN: grep -c "movx[[:space:]]*crrnd" %t.s | FileCheck %s --check-prefix=ONE
// RUN: grep -c "movx[[:space:]]*crsat," %t.s | FileCheck %s --check-prefix=ONE
// RUN: %aiecc --unified --get-core-elfs --tmpdir %t.uprj --output-dir %t.uout %s
// RUN: llvm-objdump -dr --no-show-raw-insn %t.uout/elfs_main_core_0_2/elfs_main_core_0_2.elf > %t.u.s
// RUN: FileCheck %s --input-file=%t.u.s
// RUN: grep -c "movx[[:space:]]*crrnd" %t.u.s | FileCheck %s --check-prefix=ONE
// RUN: grep -c "movx[[:space:]]*crsat," %t.u.s | FileCheck %s --check-prefix=ONE

// CHECK-LABEL: <main>:
// CHECK-DAG:   movx crrnd, #0x9
// CHECK-DAG:   movx crsat, #0x1
// CHECK-DAG:   vst.srs.4x

// ONE: {{^1$}}

module {
  aie.device(npu3) {
    %tile_0_2 = aie.tile(0, 2)
    %buf_a = aie.buffer(%tile_0_2) {sym_name = "buf_a"} : memref<512xi32>
    %buf_b = aie.buffer(%tile_0_2) {sym_name = "buf_b"} : memref<512xi8>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      scf.for %i = %c0 to %c512 step %c64 {
        %a = vector.load %buf_a[%i] : memref<512xi32>, vector<64xi32>
        %y = aievec.aie4.srs %a {shift = 12 : i32, isSigned = true, rounding = #aievec.aie4_rnd<pos_inf>, saturation = #aievec.aie4_sat<saturate>} : vector<64xi32> -> vector<64xi8>
        vector.store %y, %buf_b[%i] : memref<512xi8>, vector<64xi8>
      }
      aie.end
    }
  }
}

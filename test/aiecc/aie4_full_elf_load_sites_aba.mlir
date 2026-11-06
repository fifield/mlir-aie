// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A -> B -> A: each load site gets its own section and label (aiebu rejects
// two LOAD_PDIs of one label in a column). Both copies of @a are the same
// file and share one id.

// RUN: rm -rf aie4_full_elf_load_sites_aba_prj && mkdir -p aie4_full_elf_load_sites_aba_prj
// RUN: %aiecc --get-full-elf --dump-intermediates --tmpdir=aie4_full_elf_load_sites_aba_prj --output-dir=aie4_full_elf_load_sites_aba_prj %s
// RUN: FileCheck %s --input-file=aie4_full_elf_load_sites_aba_prj/full_elf_cert.mlir.asm
// RUN: diff aie4_full_elf_load_sites_aba_prj/a__site0.asm aie4_full_elf_load_sites_aba_prj/a__site1.asm
// RUN: FileCheck %s --check-prefix=SEC --input-file=aie4_full_elf_load_sites_aba_prj/a__site0.asm

// CHECK: LOAD_PDI [[A:[0-9]+]], @a__site0
// CHECK-NOT: [[A]]
// CHECK: LOAD_PDI {{[0-9]+}}, @b__site0
// CHECK: LOAD_PDI [[A]], @a__site1

// SEC: START_JOB 1
// SEC-NEXT: WRITE_32 0x006a3000, 0x00000002
// SEC: WRITE_32 0x006ae000, 0x00000001

module {
  aie.device(npu3_1col) @main {
    %s0 = aie.tile(0, 0)
    aie.runtime_sequence @seq(%x: memref<16xi32>) {
      aiex.configure @a {
        aiex.run @run(%x) : (memref<16xi32>)
      }
      aiex.configure @b {
        aiex.run @run(%x) : (memref<16xi32>)
      }
      aiex.configure @a {
        aiex.run @run(%x) : (memref<16xi32>)
      }
    }
  }
  aie.device(npu3_1col) @a {
    %t = aie.tile(0, 2)
    %l = aie.lock(%t, 0) {init = 1 : i32}
    aie.runtime_sequence @run(%x: memref<16xi32>) {
    }
  }
  aie.device(npu3_1col) @b {
    %t = aie.tile(0, 2)
    %l = aie.lock(%t, 1) {init = 2 : i32}
    aie.runtime_sequence @run(%x: memref<16xi32>) {
    }
  }
}

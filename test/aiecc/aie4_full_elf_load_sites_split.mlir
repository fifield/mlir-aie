// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A -> work -> A where @a's configuration (a buffer
// initializer of 4096 words) splits across pages: both sites load copies of
// one legalized template, so they share an id and the two section files are
// identical, each with several pages.

// RUN: rm -rf aie4_full_elf_load_sites_split_prj && mkdir -p aie4_full_elf_load_sites_split_prj
// RUN: %aiecc --get-full-elf --dump-intermediates --tmpdir=aie4_full_elf_load_sites_split_prj --output-dir=aie4_full_elf_load_sites_split_prj %s
// RUN: FileCheck %s --input-file=aie4_full_elf_load_sites_split_prj/full_elf_cert.mlir.asm
// RUN: diff aie4_full_elf_load_sites_split_prj/a__site0.asm aie4_full_elf_load_sites_split_prj/a__site1.asm
// RUN: FileCheck %s --check-prefix=SEC --input-file=aie4_full_elf_load_sites_split_prj/a__site0.asm

// CHECK: LOAD_PDI [[A:[0-9]+]], @a__site0
// CHECK: WRITE_32 0x00001000, 0x00005e5e
// CHECK: LOAD_PDI [[A]], @a__site1

// SEC: START_JOB 1
// SEC: .eop
// SEC: START_JOB 2
// SEC: .eop

module {
  aie.device(npu3_1col) @main {
    %s0 = aie.tile(0, 0)
    aie.runtime_sequence @seq(%x: memref<16xi32>) {
      aiex.configure @a {
        aiex.run @run(%x) : (memref<16xi32>)
      }
      %a = arith.constant 0x1000 : i32
      %v = arith.constant 0x5E5E : i32
      aiex.npu.write32(%a, %v) : i32, i32
      aiex.configure @a {
        aiex.run @run(%x) : (memref<16xi32>)
      }
    }
  }
  aie.device(npu3_1col) @a {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) {address = 0 : i32, sym_name = "big", initial_value = dense<7> : tensor<4096xi32>} : memref<4096xi32>
    aie.runtime_sequence @run(%x: memref<16xi32>) {
    }
  }
}

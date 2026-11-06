//===- aie4_section_reset.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// AIE4 configuration delivered as LOAD_PDI sections: every section starts
// with the reset of the whole partition (all three columns of npu3), and only
// then writes its device's configuration. The runtime-sequence jobs never reset anything. For A, B, A,
// both copies of @a and the copy of @b each carry the reset, so the A -> B
// and B -> A transitions (cert does not skip them: the ids differ) start
// from a quiesced array.
//
// Under --expand-load-pdis each configuration is written in line instead,
// after an aiex.npu.write_reset that writes the same reset; nothing is loaded.
// The in-line configuration zeroes lock 3 (no initializer), in the full ELF
// and in the per-sequence NPU instructions (where the lock's address,
// 0x02110030, is a word of the binary only if it is written).

// RUN: rm -rf aie4_section_reset_prj && mkdir -p aie4_section_reset_prj
// RUN: %aiecc --get-full-elf --dump-intermediates --tmpdir=aie4_section_reset_prj/on --output-dir=aie4_section_reset_prj/on %s
// RUN: FileCheck %s --check-prefix=MAIN --input-file=aie4_section_reset_prj/on/full_elf_cert.mlir.asm
// RUN: FileCheck %s --check-prefixes=SEC,CFGA --input-file=aie4_section_reset_prj/on/a__site0.asm
// RUN: FileCheck %s --check-prefixes=SEC,CFGA --input-file=aie4_section_reset_prj/on/a__site1.asm
// RUN: FileCheck %s --check-prefixes=SEC,CFGB --input-file=aie4_section_reset_prj/on/b__site0.asm

// RUN: %aiecc --get-full-elf --expand-load-pdis --dump-intermediates --tmpdir=aie4_section_reset_prj/expand --output-dir=aie4_section_reset_prj/expand %s
// RUN: FileCheck %s --check-prefix=EXP --input-file=aie4_section_reset_prj/expand/full_elf_cert.mlir.asm

// RUN: %aiecc --get-npu-insts --expand-load-pdis --tmpdir=aie4_section_reset_prj/insts_on --output-dir=aie4_section_reset_prj/insts_on %s
// RUN: od -An -tx4 -v aie4_section_reset_prj/insts_on/insts_main_seq.bin | FileCheck %s --check-prefix=INSTSON

// The program: no reset (no Core_Control write, no poll, no memtile DMA
// channel reset) in any job; the sections are included, not inlined.
// MAIN-NOT: {{MASK_POLL_32|a3000,|9e00,}}
// MAIN: LOAD_PDI [[A:[0-9]+]], @a__site0
// MAIN-NOT: {{MASK_POLL_32|a3000,|9e00,}}
// MAIN: LOAD_PDI {{[0-9]+}}, @b__site0
// MAIN-NOT: {{MASK_POLL_32|a3000,|9e00,}}
// MAIN: LOAD_PDI [[A]], @a__site1
// MAIN-NOT: {{MASK_POLL_32|a3000,|9e00,}}

// The top of every section.
// Q0: hold every core in reset, column 0 first, column 2 last.
// SEC: START_JOB 1
// SEC-NEXT: MASK_WRITE_32 {{ *}}0x006a3000, 0x00000002, 0x00000002
// SEC-NEXT: MASK_WRITE_32 {{ *}}0x006a3000, 0x00000001, 0x00000000
// SEC: MASK_WRITE_32 {{ *}}0x049a3000, 0x00000002, 0x00000002
// SEC-NEXT: MASK_WRITE_32 {{ *}}0x049a3000, 0x00000001, 0x00000000
// QB: poll every Core_Control back.
// SEC-NEXT: MASK_POLL_32 {{ *}}0x006a3000, 0x00000003, 0x00000002
// SEC: MASK_POLL_32 {{ *}}0x049a3000, 0x00000003, 0x00000002
// Q1: assert the memtile and core-tile DMA channel resets.
// SEC-NEXT: MASK_WRITE_32 {{ *}}0x00109e00, 0x00000002, 0x00000002
// SEC: MASK_WRITE_32 {{ *}}0x049ace10, 0x00000002, 0x00000002
// Q2: release them.
// SEC-NEXT: WRITE_32 {{ *}}0x00109e00, 0x00000000
// SEC: WRITE_32 {{ *}}0x049ace10, 0x00000000
// Q2p: clear the memtile MM2S pad values.
// SEC-NEXT: WRITE_32 {{ *}}0x00109f70, 0x00000000
// Q2s: return the shim data channels to their defaults.
// SEC: WRITE_32 {{ *}}0x00018540, 0x00000000
// Q3: release the core resets, leaving the cores disabled.
// SEC: MASK_WRITE_32 {{ *}}0x006a3000, 0x00000002, 0x00000000
// SEC: MASK_WRITE_32 {{ *}}0x049a3000, 0x00000002, 0x00000000
// Then the device's own configuration, and no second reset.
// CFGA-NEXT: WRITE_32 {{ *}}0x006ae000, 0x00000001
// CFGA-NEXT: WRITE_32 {{ *}}0x02110000, 0x00000005
// Memtile (1, 1) lock 3 has no initializer: zeroed, not left as it was.
// CFGA-NEXT: WRITE_32 {{ *}}0x02110030, 0x00000000
// CFGB-NEXT: WRITE_32 {{ *}}0x006ae010, 0x00000002
// SEC-NOT: {{MASK_POLL_32|a3000,|9e00,}}
// SEC: END_JOB
// SEC-NOT: {{MASK_POLL_32|a3000,|9e00,}}

// --expand-load-pdis: no load; each configuration written in line starts
// with the reset, then writes its device's configuration, zeroing lock 3.
// EXP-NOT: LOAD_PDI
// EXP: MASK_POLL_32 {{ *}}0x006a3000, 0x00000003, 0x00000002
// EXP-NOT: LOAD_PDI
// EXP: WRITE_32 {{ *}}0x006ae000, 0x00000001
// EXP-NEXT: WRITE_32 {{ *}}0x02110000, 0x00000005
// EXP-NEXT: WRITE_32 {{ *}}0x02110030, 0x00000000
// EXP: MASK_POLL_32 {{ *}}0x006a3000, 0x00000003, 0x00000002
// EXP: WRITE_32 {{ *}}0x006ae010, 0x00000002
// EXP: MASK_POLL_32 {{ *}}0x006a3000, 0x00000003, 0x00000002
// EXP: WRITE_32 {{ *}}0x006ae000, 0x00000001
// EXP-NOT: LOAD_PDI

// INSTSON: 02110030

module {
  aie.device(npu3) @main {
    %s0 = aie.tile(0, 0)
    aie.runtime_sequence @seq(%x: memref<16xi32>) {
      %a0 = arith.constant 4096 : i32
      %v0 = arith.constant 161 : i32
      aiex.npu.write32(%a0, %v0) : i32, i32
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
  aie.device(npu3) @a {
    %t = aie.tile(0, 2)
    %m = aie.tile(1, 1)
    %l = aie.lock(%t, 0) {init = 1 : i32}
    %lm = aie.lock(%m, 0) {init = 5 : i32}
    %lm3 = aie.lock(%m, 3)
    aie.runtime_sequence @run(%x: memref<16xi32>) {
      %a = arith.constant 8192 : i32
      %v = arith.constant 170 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }
  aie.device(npu3) @b {
    %t = aie.tile(0, 2)
    %l = aie.lock(%t, 1) {init = 2 : i32}
    aie.runtime_sequence @run(%x: memref<16xi32>) {
      %a = arith.constant 8192 : i32
      %v = arith.constant 187 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }
}

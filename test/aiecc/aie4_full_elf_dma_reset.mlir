//===- aie4_full_elf_dma_reset.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The configuration of an AIE4 full ELF (the entry's LOAD_PDI
// section, main__site0) starts by quiescing the whole partition -- every
// column of npu3, not just the one the design uses -- before the design's own
// lock and DMA configuration. The design is core-free (a shim -> memtile ->
// shim objectFifo link), so no kernel compiler is needed.

// RUN: rm -rf aie4_full_elf_dma_reset_prj && mkdir -p aie4_full_elf_dma_reset_prj
// RUN: %aiecc --get-full-elf --dump-intermediates --tmpdir=aie4_full_elf_dma_reset_prj/default --output-dir=aie4_full_elf_dma_reset_prj/default %s
// RUN: FileCheck %s --input-file=aie4_full_elf_dma_reset_prj/default/main__site0.asm
// RUN: FileCheck %s --check-prefix=PROG --input-file=aie4_full_elf_dma_reset_prj/default/full_elf_cert.mlir.asm

// Q0: hold core (0,2) in reset first.
// CHECK: START_JOB 1
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 {{ *}}0x006a3000, 0x00000002, 0x00000002
// CHECK-NEXT: MASK_WRITE_32 {{ *}}0x006a3000, 0x00000001, 0x00000000
// CHECK-NOT: 0x00110000,
// QB: read every core's Core_Control back, the last one in column 2.
// CHECK: MASK_POLL_32 {{ *}}0x006a3000, 0x00000003, 0x00000002
// CHECK-NOT: 0x00110000,
// CHECK: MASK_POLL_32 {{ *}}0x049a3000, 0x00000003, 0x00000002
// Q1: assert memtile DMA_S2MM_0_Ctrl's Reset, then the other columns'.
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 {{ *}}0x00109e00, 0x00000002, 0x00000002
// CHECK-NOT: 0x00110000,
// CHECK: {{^ *}}MASK_WRITE_32 {{ *}}0x04109e00, 0x00000002, 0x00000002
// CHECK-NOT: 0x00110000,
// Q2: release it.
// CHECK: {{^ *}}WRITE_32 {{ *}}0x00109e00, 0x00000000
// CHECK-NOT: 0x00110000,
// Q3: release the core resets, leaving the cores disabled.
// CHECK: {{^ *}}MASK_WRITE_32 {{ *}}0x006a3000, 0x00000002, 0x00000000
// CHECK-NOT: 0x00110000,
// CHECK: {{^ *}}MASK_WRITE_32 {{ *}}0x049a3000, 0x00000002, 0x00000000
// Only then the design's configuration: memtile lock 0, then the queue start.
// CHECK-NEXT: {{^ *}}WRITE_32 {{ *}}0x00110000, 0x00000002
// CHECK-NOT: {{MASK_POLL_32|0x00109e00,|0x006a3000,}}
// CHECK: {{^ *}}WRITE_32 {{ *}}0x00109e04, 0x00000000
// CHECK-NOT: {{MASK_POLL_32|0x00109e00,|0x006a3000,}}
// CHECK: END_JOB

// The program loads the section, and the dispatch that follows does not reset
// anything.
// PROG: START_JOB 1
// PROG-NEXT: LOAD_PDI {{ *}}{{[0-9]+}}, @main__site0
// PROG-NEXT: END_JOB
// PROG: START_JOB {{[0-9]+}}
// PROG-NOT: {{MASK_POLL_32|0x00109e00,|0x006a3000,}}
// PROG: END_JOB

module {
  aie.device(npu3) @main {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    aie.objectfifo @in(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%mem, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
    aie.runtime_sequence @seq(%a: memref<64xi32>, %b: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 0 : i64, metadata = @in, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 1 : i64, metadata = @out, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @in}
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}

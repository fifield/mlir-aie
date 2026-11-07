//===- aie4_full_elf_dma_reset_core.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// With a compiled core, the configure-time reset of the partition still comes
// first in the configure job: every phase (Q0-Q3, with the stream switch clear
// S between QB and Q1) precedes the core ELF load,
// which disables core (0,2), resets its DMA channels, writes its program
// memory (0x680000) and finally enables it.

// RUN: rm -rf aie4_full_elf_dma_reset_core_prj && mkdir -p aie4_full_elf_dma_reset_core_prj
// RUN: %aiecc --get-full-elf --dump-intermediates --tmpdir=aie4_full_elf_dma_reset_core_prj/default --output-dir=aie4_full_elf_dma_reset_core_prj/default %s
// RUN: FileCheck %s --input-file=aie4_full_elf_dma_reset_core_prj/default/main__site0.asm

// CHECK: START_JOB 1
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 {{ *}}0x006a3000, 0x00000002, 0x00000002
// CHECK-NEXT: MASK_WRITE_32 {{ *}}0x006a3000, 0x00000001, 0x00000000
// CHECK-NOT: uC_DMA_WRITE_DES_SYNC
// CHECK: MASK_POLL_32 {{ *}}0x006a3000, 0x00000003, 0x00000002
// S, split by page legalization over several jobs
// (aie4_full_elf_dma_reset.mlir), then Q1.
// CHECK: MASK_POLL_32 {{ *}}0x049a3000, 0x00000003, 0x00000002
// CHECK-NEXT: uC_DMA_WRITE_DES_SYNC {{ *}}@chain_{{[0-9]+}}
// CHECK-NOT: {{WRITE_32|POLL_32}}
// CHECK: {{^ *}}MASK_WRITE_32 {{ *}}0x00109e00, 0x00000002, 0x00000002
// CHECK-NOT: uC_DMA_WRITE_DES_SYNC
// CHECK: {{^ *}}MASK_WRITE_32 {{ *}}0x006ace00, 0x00000002, 0x00000002
// CHECK-NOT: uC_DMA_WRITE_DES_SYNC
// CHECK: {{^ *}}WRITE_32 {{ *}}0x006ace00, 0x00000000
// CHECK-NOT: uC_DMA_WRITE_DES_SYNC
// CHECK: {{^ *}}MASK_WRITE_32 {{ *}}0x006a3000, 0x00000002, 0x00000000
// CHECK-NOT: uC_DMA_WRITE_DES_SYNC
// CHECK: {{^ *}}MASK_WRITE_32 {{ *}}0x049a3000, 0x00000002, 0x00000000
// The core ELF load follows.
// CHECK-NEXT: MASK_WRITE_32 {{ *}}0x006a3000, 0x00000001, 0x00000000
// CHECK-NOT: MASK_POLL_32
// CHECK: uC_DMA_WRITE_DES_SYNC {{ *}}@[[PM:chain_[0-9]+]]
// CHECK-NOT: MASK_POLL_32
// CHECK: MASK_WRITE_32 {{ *}}0x006a3000, 0x00000001, 0x00000001
// CHECK-NEXT: END_JOB
// CHECK: [[PM]]:
// CHECK-NEXT: UC_DMA_BD {{ *}}0, 0x00680000,

module {
  aie.device(npu3) @main {
    %t02 = aie.tile(0, 2)
    %buf = aie.buffer(%t02) {sym_name = "cbuf"} : memref<16xi32>
    %core = aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %v = arith.constant 7 : i32
      memref.store %v, %buf[%c0] : memref<16xi32>
      aie.end
    }
    aie.runtime_sequence @seq(%a: memref<16xi32>) {
    }
  }
}

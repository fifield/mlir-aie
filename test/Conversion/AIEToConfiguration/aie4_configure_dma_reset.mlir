//===- aie4_configure_dma_reset.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: rm -rf aie4_configure_dma_reset_prj && split-file %s aie4_configure_dma_reset_prj
// RUN: mkdir -p aie4_configure_dma_reset_prj/v1_c1
// RUN: aie-opt --aie-assign-bd-ids --convert-aie-to-transaction --aie-npu-to-cert --cert-legalize-pages aie4_configure_dma_reset_prj/c1.mlir | aie-translate --aie-cert-to-asm --cert-section-dir=aie4_configure_dma_reset_prj/v1_c1 -o aie4_configure_dma_reset_prj/v1_c1.asm
// RUN: sed -e '/^\.include main\.asm$/r aie4_configure_dma_reset_prj/v1_c1/main.asm' -e '/^\.include main\.asm$/d' aie4_configure_dma_reset_prj/v1_c1.asm > aie4_configure_dma_reset_prj/v1_c1.inl.asm
// RUN: mkdir -p aie4_configure_dma_reset_prj/v1_c2
// RUN: aie-opt --aie-assign-bd-ids --convert-aie-to-transaction --aie-npu-to-cert --cert-legalize-pages aie4_configure_dma_reset_prj/c2.mlir | aie-translate --aie-cert-to-asm --cert-section-dir=aie4_configure_dma_reset_prj/v1_c2 -o aie4_configure_dma_reset_prj/v1_c2.asm
// RUN: sed -e '/^\.include main\.asm$/r aie4_configure_dma_reset_prj/v1_c2/main.asm' -e '/^\.include main\.asm$/d' aie4_configure_dma_reset_prj/v1_c2.asm > aie4_configure_dma_reset_prj/v1_c2.inl.asm
// RUN: mkdir -p aie4_configure_dma_reset_prj/v1_c3
// RUN: aie-opt --aie-assign-bd-ids --convert-aie-to-transaction --aie-npu-to-cert --cert-legalize-pages aie4_configure_dma_reset_prj/c3.mlir | aie-translate --aie-cert-to-asm --cert-section-dir=aie4_configure_dma_reset_prj/v1_c3 -o aie4_configure_dma_reset_prj/v1_c3.asm
// RUN: sed -e '/^\.include main\.asm$/r aie4_configure_dma_reset_prj/v1_c3/main.asm' -e '/^\.include main\.asm$/d' aie4_configure_dma_reset_prj/v1_c3.asm > aie4_configure_dma_reset_prj/v1_c3.inl.asm
// RUN: cat aie4_configure_dma_reset_prj/v1_c1.inl.asm aie4_configure_dma_reset_prj/v1_c2.inl.asm aie4_configure_dma_reset_prj/v1_c3.inl.asm | FileCheck %s

// An AIE4 configuration section starts by resetting the whole partition
// (Q0-Q3) before anything the device itself configures.

// npu3_1col: one column, shim row 0, memtile row 1, core rows 2-5. Every
// core and DMA channel in the column is reset, declared or not. Core (0,2)
// Core_Control is at 0x6a3000; column 0's memtile channels are in two groups,
// 0x109e00 (S2MM_0-3, MM2S_0-4) and 0x149e00 (S2MM_4-7, MM2S_5-9).
// CHECK-LABEL: .partition 1column
// CHECK: START_JOB {{[0-9]+}}
// Q0: every core held in reset (Reset=1, Enable=0).
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x006a3000, 0x00000002, 0x00000002
// CHECK-NEXT: MASK_WRITE_32 0x006a3000, 0x00000001, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x007a3000, 0x00000002, 0x00000002
// CHECK-NEXT: MASK_WRITE_32 0x007a3000, 0x00000001, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x008a3000, 0x00000002, 0x00000002
// CHECK-NEXT: MASK_WRITE_32 0x008a3000, 0x00000001, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x009a3000, 0x00000002, 0x00000002
// CHECK-NEXT: MASK_WRITE_32 0x009a3000, 0x00000001, 0x00000000
// QB: read each Core_Control back (mask Reset|Enable, value Reset).
// CHECK-NEXT: MASK_POLL_32 0x006a3000, 0x00000003, 0x00000002
// CHECK-NEXT: MASK_POLL_32 0x007a3000, 0x00000003, 0x00000002
// CHECK-NEXT: MASK_POLL_32 0x008a3000, 0x00000003, 0x00000002
// CHECK-NEXT: MASK_POLL_32 0x009a3000, 0x00000003, 0x00000002
// Q1: assert the reset of every memtile and core-tile channel, all S2MM then
// all MM2S per tile, unused channels included.
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x00109e00, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x00109e08, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x00109e10, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x00109e18, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x00149e00, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x00149e08, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x00149e10, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x00149e18, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x00109e60, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x00109e68, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x00109e70, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x00109e78, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x00109e80, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x00149e60, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x00149e68, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x00149e70, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x00149e78, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x00149e80, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x006ace00, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x006ace08, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x006ace10, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x007ace00, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x007ace08, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x007ace10, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x008ace00, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x008ace08, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x008ace10, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x009ace00, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x009ace08, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x009ace10, 0x00000002, 0x00000002
// Q2: the same registers, released with default modes.
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109e00, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109e08, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109e10, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109e18, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00149e00, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00149e08, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00149e10, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00149e18, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109e60, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109e68, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109e70, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109e78, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109e80, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00149e60, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00149e68, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00149e70, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00149e78, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00149e80, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x006ace00, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x006ace08, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x006ace10, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x007ace00, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x007ace08, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x007ace10, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x008ace00, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x008ace08, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x008ace10, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x009ace00, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x009ace08, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x009ace10, 0x00000000
// Q2p: every memtile MM2S channel's constant pad value cleared (both register
// groups, 0x109f70 and 0x149f70), so a previous design's pad does not leak.
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109f70, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109f74, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109f78, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109f7c, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109f80, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00149f70, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00149f74, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00149f78, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00149f7c, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00149f80, 0x00000000
// Q2s: shim data channels back to default modes. DMA_Trace_S2MM_Ctrl
// (0x18548) and DMA_Control_MM2S_0/1_Ctrl (0x18560, 0x58560) are not touched.
// CHECK-NEXT: {{^ *}}WRITE_32 0x00018540, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00058540, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00018550, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00018558, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00058550, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00058558, 0x00000000
// Q3: release the core resets, leaving the cores disabled.
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x006a3000, 0x00000002, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x007a3000, 0x00000002, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x008a3000, 0x00000002, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x009a3000, 0x00000002, 0x00000000
// The configuration proper follows: core (0,2)'s reset pulse and lock zeroing
// from initLocks, unchanged.
// CHECK-NEXT: MASK_WRITE_32 0x006a3000, 0x00000002, 0x00000002
// CHECK-NEXT: MASK_WRITE_32 0x006a3000, 0x00000002, 0x00000000
// CHECK-NOT: {{^ *}}WRITE_32 0x006a3000, 0x00000001
// CHECK-NOT: {{(WRITE|POLL)_32 +0x0[2-7]}}
// CHECK-NOT: {{^ *}}WRITE_32 0x00110{{0[2-9a-f]|[1-3][0-9a-f]}}0,
// CHECK-NOT: {{0x00018548|0x00018560|0x00058560}}
// Declared memtile locks: lock 0 is initialized to 1, and lock 1, declared
// without an initializer, to 0. Undeclared lock IDs are left alone.
// CHECK: {{^ *}}WRITE_32 0x00110000, 0x00000001
// CHECK-NEXT: {{^ *}}WRITE_32 0x00110010, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x006ae000, 0x00000001
// CHECK-NOT: {{^ *}}WRITE_32 0x006a3000, 0x00000001
// CHECK-NOT: {{(WRITE|POLL)_32 +0x0[2-7]}}
// CHECK-NOT: {{^ *}}WRITE_32 0x00110{{0[2-9a-f]|[1-3][0-9a-f]}}0,
// CHECK-NOT: {{0x00018548|0x00018560|0x00058560}}
// The first START_QUEUE comes after the quiesce.
// CHECK: {{^ *}}WRITE_32 0x00109e04, 0x00000000
// Core (0,2) has no elf_file and is not enabled.
// CHECK-NOT: {{^ *}}WRITE_32 0x006a3000, 0x00000001
// CHECK-NOT: {{(WRITE|POLL)_32 +0x0[2-7]}}
// CHECK-NOT: {{^ *}}WRITE_32 0x00110{{0[2-9a-f]|[1-3][0-9a-f]}}0,
// CHECK-NOT: {{0x00018548|0x00018560|0x00058560}}
// CHECK: END_JOB
//--- c1.mlir
module {
  aie.device(npu3_1col) @main {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %buf0 = aie.buffer(%t01) {address = 0 : i32} : memref<16xi32>
    %prod = aie.lock(%t01, 0) {init = 1 : i32}
    %cons = aie.lock(%t01, 1)
    aie.memtile_dma(%t01) {
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^mm2s)
    ^bd0:
      %c0 = arith.constant 1 : i32
      aie.use_lock(%prod, AcquireGreaterEqual, %c0)
      aie.dma_bd(%buf0 : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
      aie.use_lock(%cons, Release, %c0)
      aie.next_bd ^bd0
    ^mm2s:
      %1 = aie.dma_start(MM2S, 0, ^bd1, ^end)
    ^bd1:
      %c1 = arith.constant 1 : i32
      aie.use_lock(%cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf0 : memref<16xi32> offset = 0 len = 16) {bd_id = 1 : i32}
      aie.use_lock(%prod, Release, %c1)
      aie.next_bd ^bd1
    ^end:
      aie.end
    }
    %buf2 = aie.buffer(%t02) {address = 1024 : i32} : memref<16xi32>
    %l2 = aie.lock(%t02, 0) {init = 1 : i32}
    aie.mem(%t02) {
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      %c0 = arith.constant 1 : i32
      aie.use_lock(%l2, AcquireGreaterEqual, %c0)
      aie.dma_bd(%buf2 : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
      aie.use_lock(%l2, Release, %c0)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.core(%t02) {
      aie.end
    }
  }
}

// -----
//--- c2.mlir

// npu3_2col: the quiesce covers both columns of the partition although the
// device declares only column 0. Column c's addresses add c << 25, and each
// phase finishes column 0 before column 1.
// CHECK-LABEL: .partition 2column
// CHECK: START_JOB {{[0-9]+}}
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x006a3000, 0x00000002, 0x00000002
// CHECK-NEXT: MASK_WRITE_32 0x006a3000, 0x00000001, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x007a3000, 0x00000002, 0x00000002
// CHECK-NEXT: MASK_WRITE_32 0x007a3000, 0x00000001, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x008a3000, 0x00000002, 0x00000002
// CHECK-NEXT: MASK_WRITE_32 0x008a3000, 0x00000001, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x009a3000, 0x00000002, 0x00000002
// CHECK-NEXT: MASK_WRITE_32 0x009a3000, 0x00000001, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x026a3000, 0x00000002, 0x00000002
// CHECK-NEXT: MASK_WRITE_32 0x026a3000, 0x00000001, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x027a3000, 0x00000002, 0x00000002
// CHECK-NEXT: MASK_WRITE_32 0x027a3000, 0x00000001, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x028a3000, 0x00000002, 0x00000002
// CHECK-NEXT: MASK_WRITE_32 0x028a3000, 0x00000001, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x029a3000, 0x00000002, 0x00000002
// CHECK-NEXT: MASK_WRITE_32 0x029a3000, 0x00000001, 0x00000000
// CHECK-NEXT: MASK_POLL_32 0x006a3000, 0x00000003, 0x00000002
// CHECK:      MASK_POLL_32 0x029a3000, 0x00000003, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x00109e00, 0x00000002, 0x00000002
// CHECK:      {{^ *}}MASK_WRITE_32 0x009ace10, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x02109e00, 0x00000002, 0x00000002
// CHECK:      {{^ *}}MASK_WRITE_32 0x02149e80, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x026ace00, 0x00000002, 0x00000002
// CHECK:      {{^ *}}MASK_WRITE_32 0x029ace10, 0x00000002, 0x00000002
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109e00, 0x00000000
// CHECK:      {{^ *}}WRITE_32 0x029ace10, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109f70, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109f74, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109f78, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109f7c, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109f80, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00149f70, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00149f74, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00149f78, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00149f7c, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00149f80, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x02109f70, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x02109f74, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x02109f78, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x02109f7c, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x02109f80, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x02149f70, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x02149f74, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x02149f78, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x02149f7c, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x02149f80, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00018540, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00058540, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00018550, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00018558, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00058550, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x00058558, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x02018540, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x02058540, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x02018550, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x02018558, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x02058550, 0x00000000
// CHECK-NEXT: {{^ *}}WRITE_32 0x02058558, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x006a3000, 0x00000002, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x007a3000, 0x00000002, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x008a3000, 0x00000002, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x009a3000, 0x00000002, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x026a3000, 0x00000002, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x027a3000, 0x00000002, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x028a3000, 0x00000002, 0x00000000
// CHECK-NEXT: {{^ *}}MASK_WRITE_32 0x029a3000, 0x00000002, 0x00000000
// CHECK-NOT: {{(WRITE|POLL)_32 +0x0[4-7]}}
// CHECK: END_JOB
module {
  aie.device(npu3_2col) @main {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %buf0 = aie.buffer(%t01) {address = 0 : i32} : memref<16xi32>
    %l0 = aie.lock(%t01, 0) {init = 1 : i32}
    aie.memtile_dma(%t01) {
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      %c1 = arith.constant 1 : i32
      aie.use_lock(%l0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf0 : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
      aie.use_lock(%l0, Release, %c1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}

// -----
//--- c3.mlir

// Out-of-order S2MM: Q2 leaves memtile DMA_S2MM_1_Ctrl (0x109e08) at its
// default, and the configuration then writes the channel's requested
// Enable_Out_of_Order (bit 3) before its START_QUEUE. The in-order MM2S_0
// (0x109e60) gets no channel-control write after Q2.
// CHECK-LABEL: .partition 1column
// CHECK: START_JOB {{[0-9]+}}
// CHECK: {{^ *}}MASK_WRITE_32 0x00109e08, 0x00000002, 0x00000002
// CHECK: {{^ *}}MASK_WRITE_32 0x00109e60, 0x00000002, 0x00000002
// CHECK: {{^ *}}WRITE_32 0x00109e08, 0x00000000
// CHECK: {{^ *}}WRITE_32 0x00109e60, 0x00000000
// CHECK: {{^ *}}MASK_WRITE_32 0x009a3000, 0x00000002, 0x00000000
// CHECK-NOT: 0x00109e60,
// CHECK: {{^ *}}WRITE_32 0x00109e08, 0x00000008
// CHECK-NEXT: {{^ *}}WRITE_32 0x00109e0c, 0x00010000
// CHECK-NOT: 0x00109e60,
// CHECK: {{^ *}}WRITE_32 0x00109e64, 0x00000002
// CHECK-NOT: 0x00109e60,
// CHECK: END_JOB
module {
  aie.device(npu3_1col) @main {
    %t = aie.tile(0, 1)
    %b0 = aie.buffer(%t) {address = 0 : i32} : memref<8xi32>
    %b1 = aie.buffer(%t) {address = 64 : i32} : memref<8xi32>
    %l = aie.lock(%t, 0) {init = 0 : i32}
    aie.memtile_dma(%t) {
      %0 = aie.dma_start(S2MM, 1, ^bd0, ^mm2s, repeat_count = 1) {out_of_order}
    ^bd0:
      %c1 = arith.constant 1 : i32
      aie.dma_bd(%b0 : memref<8xi32> offset = 0 len = 8) {bd_id = 0 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(%l, Release, %c1)
      aie.next_bd ^bd1
    ^bd1:
      %c1b = arith.constant 1 : i32
      aie.dma_bd(%b1 : memref<8xi32> offset = 0 len = 8) {bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
      aie.use_lock(%l, Release, %c1b)
      aie.next_bd ^bd0
    ^mm2s:
      %1 = aie.dma_start(MM2S, 0, ^bd2, ^end)
    ^bd2:
      %c1c = arith.constant 1 : i32
      aie.use_lock(%l, AcquireGreaterEqual, %c1c)
      aie.dma_bd(%b0 : memref<8xi32> offset = 0 len = 8) {bd_id = 2 : i32}
      aie.next_bd ^bd2
    ^end:
      aie.end
    }
  }
}

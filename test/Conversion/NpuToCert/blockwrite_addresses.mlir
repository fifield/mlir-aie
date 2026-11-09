// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-npu-to-cert --split-input-file %s | FileCheck %s --implicit-check-not=aiex.npu.blockwrite
// RUN: aie-opt --aie-dma-to-npu --aie-npu-to-cert --split-input-file %s | FileCheck %s --implicit-check-not=aiex.npu.blockwrite
// RUN: aie-opt --aie-npu-to-cert --split-input-file %s | aie-translate --aie-cert-to-asm --split-input-file | FileCheck %s --check-prefix=ASM

// NPU3 logical core row 2 maps to address row 6. Tile coordinates replace
// existing upper address bits; an address without coordinates stays absolute.
// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.uc_dma_bd @data, 39846144, 4, false
// CHECK: aiex.cert.uc_dma_bd @data, 39846144, 4, false
// CHECK: aiex.cert.uc_dma_bd @data, 305398016, 4, false
// ASM: UC_DMA_BD 0, 0x02600100, @data, 4, 0, 0
// ASM: UC_DMA_BD 0, 0x02600100, @data, 4, 0, 0
// ASM: UC_DMA_BD 0, 0x12340100, @data, 4, 0, 0
aie.device(npu3) {
  memref.global "private" constant @data : memref<4xi32> = dense<[1, 2, 3, 4]>
  aie.runtime_sequence @seq() {
    %data = memref.get_global @data : memref<4xi32>
    aiex.npu.blockwrite(%data) {address = 0x100 : ui32, column = 1 : i32, row = 2 : i32} : memref<4xi32>
    aiex.npu.blockwrite(%data) {address = 0x12300100 : ui32, column = 1 : i32, row = 2 : i32} : memref<4xi32>
    aiex.npu.blockwrite(%data) {address = 0x12340100 : ui32} : memref<4xi32>
  }
}

// -----

// NPU2 uses physical row 2 directly, so the same coordinates resolve to a
// different tile base. Buffer offsets are words, not bytes.
// CHECK-LABEL: aie.device(npu2)
// CHECK: aiex.cert.uc_dma_bd @data, 35651840, 4, false
// CHECK: aiex.cert.uc_dma_bd @data, 35652612, 4, false
// ASM: UC_DMA_BD 0, 0x02200100, @data, 4, 0, 0
// ASM: UC_DMA_BD 0, 0x02200404, @data, 4, 0, 0
aie.device(npu2) {
  %tile = aie.tile(1, 2)
  %buffer = aie.buffer(%tile) {address = 1024 : i32, sym_name = "buffer"} : memref<16xi32>
  memref.global "private" constant @data : memref<4xi32> = dense<7>
  aie.runtime_sequence @seq() {
    %data = memref.get_global @data : memref<4xi32>
    aiex.npu.blockwrite(%data) {address = 0x100 : ui32, column = 1 : i32, row = 2 : i32} : memref<4xi32>
    aiex.npu.blockwrite(%data) {address = 1 : ui32, buffer = @buffer} : memref<4xi32>
  }
}

// -----

// 4096 words split recursively into four 1024-word transfers. Every child
// retains the tile base and advances its destination by its byte size.
// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 39846144, 1024, false
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 39850240, 1024, false
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 39854336, 1024, false
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 39858432, 1024, false
// ASM: UC_DMA_BD 0, 0x02600100, @{{[^,]+}}, 1024, 0, 0
// ASM: UC_DMA_BD 0, 0x02601100, @{{[^,]+}}, 1024, 0, 0
// ASM: UC_DMA_BD 0, 0x02602100, @{{[^,]+}}, 1024, 0, 0
// ASM: UC_DMA_BD 0, 0x02603100, @{{[^,]+}}, 1024, 0, 0
aie.device(npu3) {
  memref.global "private" constant @data : memref<4096xi32> = dense<7>
  aie.runtime_sequence @seq() {
    %data = memref.get_global @data : memref<4096xi32>
    aiex.npu.blockwrite(%data) {address = 0x100 : ui32, column = 1 : i32, row = 2 : i32} : memref<4096xi32>
  }
}

// -----

// Splitting an absolute address must preserve all of its upper bits.
// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 305398016, 1024, false
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 305402112, 1024, false
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 305406208, 1024, false
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 305410304, 1024, false
// ASM: UC_DMA_BD 0, 0x12340100, @{{[^,]+}}, 1024, 0, 0
// ASM: UC_DMA_BD 0, 0x12341100, @{{[^,]+}}, 1024, 0, 0
// ASM: UC_DMA_BD 0, 0x12342100, @{{[^,]+}}, 1024, 0, 0
// ASM: UC_DMA_BD 0, 0x12343100, @{{[^,]+}}, 1024, 0, 0
aie.device(npu3) {
  memref.global "private" constant @data : memref<4096xi32> = dense<7>
  aie.runtime_sequence @seq() {
    %data = memref.get_global @data : memref<4096xi32>
    aiex.npu.blockwrite(%data) {address = 0x12340100 : ui32} : memref<4096xi32>
  }
}

// -----

// AIE4 memtile buffers use an additive data-memory window that can exceed
// 20 bits. Resolve the word offset and window once, including before splitting.
// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.uc_dma_bd @small, 37224452, 4, false
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 37224452, 1024, false
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 37228548, 1024, false
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 37232644, 1024, false
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 37236740, 1024, false
// ASM: UC_DMA_BD 0, 0x02380004, @small, 4, 0, 0
// ASM: UC_DMA_BD 0, 0x02380004, @{{[^,]+}}, 1024, 0, 0
// ASM: UC_DMA_BD 0, 0x02381004, @{{[^,]+}}, 1024, 0, 0
// ASM: UC_DMA_BD 0, 0x02382004, @{{[^,]+}}, 1024, 0, 0
// ASM: UC_DMA_BD 0, 0x02383004, @{{[^,]+}}, 1024, 0, 0
aie.device(npu3) {
  %tile = aie.tile(1, 1)
  %buffer = aie.buffer(%tile) {address = 0x180000 : i32, sym_name = "buffer"} : memref<4097xi32>
  memref.global "private" constant @small : memref<4xi32> = dense<7>
  memref.global "private" constant @large : memref<4096xi32> = dense<7>
  aie.runtime_sequence @seq() {
    %small = memref.get_global @small : memref<4xi32>
    %large = memref.get_global @large : memref<4096xi32>
    aiex.npu.blockwrite(%small) {address = 1 : ui32, buffer = @buffer} : memref<4xi32>
    aiex.npu.blockwrite(%large) {address = 1 : ui32, buffer = @buffer} : memref<4096xi32>
  }
}

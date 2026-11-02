// RUN: aie-opt --aie-npu-to-cert -split-input-file %s | FileCheck %s --implicit-check-not=memref.get_global --implicit-check-not=aiex.npu.blockwrite
// RUN: aie-opt --aie-npu-to-cert -split-input-file %s | aie-translate --aie-cert-to-asm -split-input-file | FileCheck %s --check-prefix=ASM

// An empty copy must not create a chain, descriptor, or DMA submission.
// CHECK-LABEL: aie.device(npu3)
// CHECK-NOT: aiex.cert.uc_dma
// CHECK: aiex.cert.job(2)
// CHECK-NOT: aiex.cert.uc_dma
// ASM: START_JOB 2
// ASM-NOT: uC_DMA_WRITE_DES_SYNC
// ASM: END_JOB
// ASM-NOT: UC_DMA_BD
aie.device(npu3) {
  memref.global "private" constant @empty : memref<0xi32> = dense<>
  aie.runtime_sequence @seq() {
    %empty = memref.get_global @empty : memref<0xi32>
    aiex.npu.blockwrite(%empty) {address = 4096 : ui32} : memref<0xi32>
  }
}

// -----

// Empty copies before, between, and after real copies must not consume chain
// IDs, reorder the real transfers, or leave dead get_global operations.
// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.uc_dma_chain @chain_0 {
// CHECK-NEXT: aiex.cert.uc_dma_bd @first, 4096, 1, false
// CHECK: aiex.cert.uc_dma_chain @chain_1 {
// CHECK-NEXT: aiex.cert.uc_dma_bd @second, 8192, 2, false
// CHECK-NOT: aiex.cert.uc_dma_chain
// CHECK: aiex.cert.job(2) {
// CHECK-NEXT: aiex.cert.uc_dma_write_des_sync(@chain_0)
// CHECK-NEXT: aiex.cert.uc_dma_write_des_sync(@chain_1)
// CHECK-NOT: aiex.cert.uc_dma
// ASM: START_JOB 2
// ASM-NEXT: uC_DMA_WRITE_DES_SYNC @chain_0
// ASM-NEXT: uC_DMA_WRITE_DES_SYNC @chain_1
// ASM-NEXT: END_JOB
// ASM: UC_DMA_BD 0, 0x00001000, @first, 1, 0, 0
// ASM: UC_DMA_BD 0, 0x00002000, @second, 2, 0, 0
// ASM-NOT: UC_DMA_BD
aie.device(npu3) {
  memref.global "private" constant @empty : memref<0xi32> = dense<>
  memref.global "private" constant @first : memref<1xi32> = dense<7>
  memref.global "private" constant @second : memref<2xi32> = dense<[8, 9]>
  aie.runtime_sequence @seq() {
    %empty = memref.get_global @empty : memref<0xi32>
    %first = memref.get_global @first : memref<1xi32>
    %second = memref.get_global @second : memref<2xi32>
    aiex.npu.blockwrite(%empty) {address = 0 : ui32} : memref<0xi32>
    aiex.npu.blockwrite(%first) {address = 4096 : ui32} : memref<1xi32>
    aiex.npu.blockwrite(%empty) {address = 0 : ui32} : memref<0xi32>
    aiex.npu.blockwrite(%second) {address = 8192 : ui32} : memref<2xi32>
    aiex.npu.blockwrite(%empty) {address = 0 : ui32} : memref<0xi32>
  }
}

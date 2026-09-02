//===- cert_isa_ops.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s
// RUN: aie-opt %s -mlir-print-op-generic | aie-opt | FileCheck %s

module {
  aie.device(npu2) {
    memref.global "private" constant @dma_data_0 : memref<9xi32> = dense<[0xa8, 0xa7, 0xa6, 0xa5, 0xa4, 0xa3, 0xa2, 0xa1, 0xa0]>
    // CHECK: aiex.cert.uc_dma_chain @dma_chain
    aiex.cert.uc_dma_chain @dma_chain {
      // CHECK: aiex.cert.uc_dma_bd @dma_data_0, 6292480, 128, true
      aiex.cert.uc_dma_bd @dma_data_0, 0x600400, 128, true
    }

    // CHECK: aiex.cert.job(1)
    aiex.cert.job(1) {
      // CHECK: aiex.cert.uc_dma_write_des(2, @dma_chain)
      aiex.cert.uc_dma_write_des(2, @dma_chain)
      // CHECK: aiex.cert.wait_uc_dma(2)
      aiex.cert.wait_uc_dma(2)
      // CHECK: aiex.cert.read32(1, 34603008)
      aiex.cert.read32(1, 34603008)
      // CHECK: aiex.cert.read32_d(0, 1)
      aiex.cert.read32_d(0, 1)
      // CHECK: aiex.cert.mov(0, 16)
      aiex.cert.mov(0, 16)
      // CHECK: aiex.cert.add(0, 1)
      aiex.cert.add(0, 1)
      // CHECK: aiex.cert.yield
      aiex.cert.yield
      // CHECK: aiex.cert.write32_d(68157440, 5)
      aiex.cert.write32_d(68157440, 5)
      // CHECK: aiex.cert.write32_d(3, 5) {address_is_reg}
      aiex.cert.write32_d(3, 5) {address_is_reg}
      // CHECK: aiex.cert.write32_d(68157440, 4) {value_is_reg}
      aiex.cert.write32_d(68157440, 4) {value_is_reg}
      // CHECK: aiex.cert.write32_d(3, 4) {address_is_reg, value_is_reg}
      aiex.cert.write32_d(3, 4) {address_is_reg, value_is_reg}
      // CHECK: aiex.cert.poll32(34603008, 1)
      aiex.cert.poll32(34603008, 1)
      // CHECK: aiex.cert.maskpoll32(34603008, 15, 1)
      aiex.cert.maskpoll32(34603008, 15, 1)
      // CHECK: aiex.cert.sleep(100)
      aiex.cert.sleep(100)
      // CHECK: aiex.cert.save_timestamps(7)
      aiex.cert.save_timestamps(7)
      // CHECK: aiex.cert.save_register(34603008, 9)
      aiex.cert.save_register(34603008, 9)
    }
  }
}

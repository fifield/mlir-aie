//===- cert_isa_ops_assemble.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This is the tier that matters: it proves the emitted text is accepted by the
// real assembler, the only ground truth for CERT syntax.

// REQUIRES: aiebu
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: aie-translate %s -aie-cert-to-asm -o %t.d/ctrl.asm
// RUN: %aiebu_asm -t aie2ps -c %t.d/ctrl.asm -o %t.d/ctrl.elf
// RUN: llvm-readelf -S %t.d/ctrl.elf | FileCheck %s
// CHECK: .ctrltext.0.0

module {
  aie.device(xcve3858) {
    memref.global "private" constant @data0 : memref<8xi32> = dense<0>
    aiex.cert.uc_dma_chain @chain0 {
      aiex.cert.uc_dma_bd @data0, 0x001a05c0, 8, false
    }
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des(2, @chain0)
      aiex.cert.wait_uc_dma(2)
      aiex.cert.read32(1, 0x02100000)
      aiex.cert.read32_d(0, 1)
      aiex.cert.mov(0, 16)
      aiex.cert.add(0, 1)
      aiex.cert.yield
      aiex.cert.write32_d(68157440, 5)
      aiex.cert.write32_d(3, 5) {address_is_reg}
      aiex.cert.write32_d(68157440, 4) {value_is_reg}
      aiex.cert.write32_d(3, 4) {address_is_reg, value_is_reg}
      aiex.cert.poll32(0x02100000, 1)
      aiex.cert.maskpoll32(0x02100000, 0xf, 1)
      aiex.cert.sleep(100)
      aiex.cert.save_timestamps(7)
      aiex.cert.save_register(0x02100000, 9)
    }
  }
}

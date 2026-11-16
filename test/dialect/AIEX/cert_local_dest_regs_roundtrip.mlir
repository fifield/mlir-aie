//===- cert_local_dest_regs_roundtrip.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// r7, the highest job-local register, is accepted wherever an instruction
// writes a register or names a uC DMA wait handle.

// RUN: aie-opt %s | FileCheck %s
// RUN: aie-opt %s | aie-opt | FileCheck %s

// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.read32(7, 34603008)
// CHECK-NEXT: aiex.cert.read32_d(23, 7)
// CHECK-NEXT: aiex.cert.uc_dma_write_des(7, @chain)
// CHECK-NEXT: aiex.cert.wait_uc_dma(7)
aie.device(npu3) {
  memref.global "private" constant @data : memref<1xi32> = dense<1>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @data, 0x1000, 1, false
  }
  aiex.cert.job(0) {
    aiex.cert.read32(7, 0x02100000)
    aiex.cert.read32_d(23, 7)
    aiex.cert.uc_dma_write_des(7, @chain)
    aiex.cert.wait_uc_dma(7)
  }
}

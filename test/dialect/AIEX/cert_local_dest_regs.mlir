//===- cert_local_dest_regs.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A register an instruction writes, and a uC DMA wait handle, must be a
// job-local register r0..r7. The firmware hangs on r8 and above. Source
// registers may still be any of r0..r23.

// RUN: aie-opt %s -split-input-file -verify-diagnostics

aie.device(npu3) {
  aiex.cert.job(0) {
    // expected-error@+1 {{attribute 'value' failed to satisfy constraint}}
    aiex.cert.read32(8, 0x02100000)
  }
}

// -----

aie.device(npu3) {
  aiex.cert.job(0) {
    // expected-error@+1 {{attribute 'value' failed to satisfy constraint}}
    aiex.cert.read32_d(0, 8)
  }
}

// -----

aie.device(npu3) {
  memref.global "private" constant @data : memref<1xi32> = dense<1>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @data, 0x1000, 1, false
  }
  aiex.cert.job(0) {
    // expected-error@+1 {{attribute 'wait_handle' failed to satisfy constraint}}
    aiex.cert.uc_dma_write_des(8, @chain)
  }
}

// -----

aie.device(npu3) {
  aiex.cert.job(0) {
    // expected-error@+1 {{attribute 'wait_handle' failed to satisfy constraint}}
    aiex.cert.wait_uc_dma(8)
  }
}

// -----

// The read32_d address is a source register and may be shared.
aie.device(npu3) {
  aiex.cert.job(0) {
    aiex.cert.read32_d(23, 7)
  }
}

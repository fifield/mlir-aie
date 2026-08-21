//===- page_cost_chain_dedup_diag.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Control for page_cost_chain_dedup.mlir: counting a re-referenced uC-DMA
// chain once must not slide into under-counting it. A genuinely oversized page
// still has to be rejected.
//
// Ground truth: aiebu-asm rejects the emitted asm for this design with "text
// and data section size 8480 > pagesize(8192)". The estimate below is 8484 --
// 4 bytes conservative, on the safe side, and agreeing on the verdict. The
// same design at 2000 words instead of 2100 assembles, and the estimator
// accepts it (page_cost_chain_dedup.mlir covers the accepting direction).

// RUN: aie-opt -cert-legalize-pages --verify-diagnostics %s

// Estimate = 32 (header) + 8 (START_JOB) + 8 + 8 (two uc_dma_write_des)
// + 4 + 4 (two wait_uc_dma) + (16 + 2100*4 = 8416 data, counted once)
// + 4 (END_JOB) = 8484. Job 1 carries a local wait handle, so the page also
// offers no legal cut and the local-register diagnostic is the one that fires.
module {
  aie.device(npu2) {
    memref.global "private" constant @dbig : memref<2100xi32> = dense<0>
    aiex.cert.uc_dma_chain @cbig {
      aiex.cert.uc_dma_bd @dbig, 0, 2100, false
    }
    // expected-error@+1 {{cert.page is an estimated 8484 bytes, over the 8192-byte microcontroller page limit}}
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.uc_dma_write_des(0, @cbig)
        aiex.cert.wait_uc_dma(0)
        aiex.cert.uc_dma_write_des(0, @cbig)
        aiex.cert.wait_uc_dma(0)
      }
    }
  }
}

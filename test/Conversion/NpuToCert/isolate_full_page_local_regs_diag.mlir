//===- isolate_full_page_local_regs_diag.mlir -------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// IsolateFullPageOpsPattern rebuilds a job that mixes cert.load_pdi or
// cert.preempt with other operations as up to three DISTINCT jobs. Registers
// r0..r7 are job-private (isa-spec.yaml), so a local value written before the
// full-page op and read after it is destroyed by that rewrite -- exactly the
// hazard SplitCertPageOpPattern avoids by declining to cut. Isolation is
// mandatory ("load_pdi and preempt should take one whole job"), so there is
// nothing to decline: the pass must reject loudly instead of silently
// producing jobs that read a fresh register.
//
// See isolate_full_page_local_regs.mlir for the controls that must still
// isolate.

// RUN: aie-opt -cert-legalize-pages --split-input-file --verify-diagnostics %s

// Case 1: r0 written before cert.load_pdi, read after it.
module {
  aie.device(npu2) {
    aiex.cert.section @config {
      aiex.cert.page { aiex.cert.job(9) { aiex.cert.write32(0x2000, 20) } }
    }
    aiex.cert.page {
      // expected-error@+1 {{job 1 mixes cert.load_pdi with other operations and must be broken into separate jobs, but register r0 is written before it and read after it}}
      aiex.cert.job(1) {
        aiex.cert.mov(0, 68157440)
        aiex.cert.load_pdi(1, @config)
        aiex.cert.write32_d(0, 7) {address_is_reg}
      }
    }
  }
}

// -----

// Case 2: the same hazard around cert.preempt, with the crossing value carried
// by a uC-DMA wait handle rather than a mov.
module {
  aie.device(npu2) {
    memref.global "private" constant @d1 : memref<4xi32> = dense<0>
    aiex.cert.uc_dma_chain @c1 {
      aiex.cert.uc_dma_bd @d1, 0, 4, false
    }
    aiex.cert.section @save {
      aiex.cert.page { aiex.cert.job(10) { aiex.cert.write32(0x2100000, 0) } }
    }
    aiex.cert.section @restore {
      aiex.cert.page { aiex.cert.job(11) { aiex.cert.write32(0x2100000, 1) } }
    }
    aiex.cert.page {
      // expected-error@+1 {{job 1 mixes cert.preempt with other operations and must be broken into separate jobs, but register r3 is written before it and read after it}}
      aiex.cert.job(1) {
        aiex.cert.uc_dma_write_des(3, @c1)
        aiex.cert.preempt(0, @save, @restore)
        aiex.cert.wait_uc_dma(3)
      }
    }
  }
}

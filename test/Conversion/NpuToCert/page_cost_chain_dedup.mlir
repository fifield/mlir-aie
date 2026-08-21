//===- page_cost_chain_dedup.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// The emitter writes one labelled block per uc_dma_chain and per payload
// global no matter how many instructions name it, and the assembler sizes the
// page from those labels. Re-referencing a chain therefore costs only the
// instruction. estimateCost used to charge the whole chain again on every
// reference, which inflated pages enough to reject ones aiebu-asm accepts.
//
// Ground truth for case 1, measured with aiebu-asm on the emitted asm:
// .ctrltext 0x40 + one copy of the data. Growing the payload with two
// references, aiebu-asm accepts 2000 words and rejects 2100 ("text and data
// section size 8480 > pagesize(8192)") -- the crossover for ONE copy of the
// payload, not two.

// See page_cost_chain_dedup_diag.mlir for the control proving dedup did not
// turn into under-counting.

// RUN: aie-opt -cert-legalize-pages --split-input-file %s | FileCheck %s --check-prefix=REUSE
// RUN: aie-opt -cert-legalize-pages --split-input-file %s | FileCheck %s --check-prefix=SHARED

// Case 1: one 1200-word chain referenced twice. Counted once the estimate is
// 32 (header) + 8 (START_JOB) + 8 + 8 (two uc_dma_write_des) + 4 + 4 (two
// wait_uc_dma) + (16 + 1200*4 = 4816 data) + 4 (END_JOB) = 4884, under the
// 8192 limit. Double-counting made it 9700 and the page was rejected as
// oversized-and-unsplittable (job 1 holds a local wait handle, so it offers no
// legal cut). Assert the page survives intact, with both references in one job.
// REUSE: aiex.cert.page
// REUSE-NEXT: aiex.cert.job(1)
// REUSE-NEXT: aiex.cert.uc_dma_write_des(0, @c1)
// REUSE-NEXT: aiex.cert.wait_uc_dma(0)
// REUSE-NEXT: aiex.cert.uc_dma_write_des(0, @c1)
// REUSE-NEXT: aiex.cert.wait_uc_dma(0)
module {
  aie.device(npu2) {
    memref.global "private" constant @d1 : memref<1200xi32> = dense<0>
    aiex.cert.uc_dma_chain @c1 {
      aiex.cert.uc_dma_bd @d1, 0, 1200, false
    }
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.uc_dma_write_des(0, @c1)
        aiex.cert.wait_uc_dma(0)
        aiex.cert.uc_dma_write_des(0, @c1)
        aiex.cert.wait_uc_dma(0)
      }
    }
  }
}

// -----

// Case 2: two DISTINCT chains pointing at one shared payload global. The BDs
// are separate labels and each still costs 16 bytes; the global they share is
// laid down once. So chains and payloads must be deduplicated separately --
// keying only on the chain would double-count @shared, keying only on the
// payload would drop a BD.
// SHARED: aiex.cert.uc_dma_write_des_sync(@ca)
// SHARED-NEXT: aiex.cert.uc_dma_write_des_sync(@cb)
module {
  aie.device(npu2) {
    memref.global "private" constant @shared : memref<1500xi32> = dense<0>
    aiex.cert.uc_dma_chain @ca {
      aiex.cert.uc_dma_bd @shared, 0, 1500, false
    }
    aiex.cert.uc_dma_chain @cb {
      aiex.cert.uc_dma_bd @shared, 4096, 1500, false
    }
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.uc_dma_write_des_sync(@ca)
        aiex.cert.uc_dma_write_des_sync(@cb)
      }
    }
  }
}

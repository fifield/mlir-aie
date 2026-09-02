//===- page_split_local_regs.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// isa-spec.yaml -- "Register r0..r7 are local registers which are private
// to each job." SplitCertPageOpPattern rebuilds the two halves of a split as
// two DISTINCT cert.job ops, so a local-register value produced on one side
// does not survive the split -- unlike the backward TCT/memory dependencies
// blessed at AIENpuToCert.cpp, which live outside the job-private
// register file. A job that touches any local register (r0..r7) is therefore
// not a split candidate at all; only jobs restricted to global registers
// (r8..r23) may be cut.

// RUN: aie-opt -cert-legalize-pages --split-input-file %s | FileCheck %s --check-prefix=MOVADD
// RUN: aie-opt -cert-legalize-pages --split-input-file %s | FileCheck %s --check-prefix=UCDMA
// RUN: aie-opt -cert-legalize-pages --split-input-file %s | FileCheck %s --check-prefix=GLOBAL

// Case 1: a mov/add pair on a LOCAL register (r0) straddling the natural split
// point. Estimate = 32 (header) + 8 (START_JOB) + 8 (mov) + 4 (uc_dma_write_des_sync)
// + (16 + 2000*4 = 8016 data) + 8 (add) + 4 (END_JOB) = 8080: over the 8000
// split trigger, under the 8192 hard limit. Without the local-register gate
// the splitter would cut between uc_dma_write_des_sync and add. Assert one
// page holding all three ops.
// Note on FileCheck scoping: --split-input-file feeds ALL three cases'
// rewritten output to a single FileCheck invocation per RUN line, so a
// directive's very first (non -NEXT) match can land in a *different* case's
// output. This directive set is for the first case, so its first page match
// is safely this case's; the rest of the sequence uses -NEXT to pin strict
// adjacency (mov/sync/add all inside the same, unsplit job), which is by
// itself sufficient to prove no split fell between them -- no NOT-style
// directive needed (a trailing one would incorrectly match the pages that
// belong to the later cases, in the same concatenated stream).
// MOVADD: aiex.cert.page
// MOVADD-NEXT: aiex.cert.job(1)
// MOVADD-NEXT: aiex.cert.mov(0, 16)
// MOVADD-NEXT: aiex.cert.uc_dma_write_des_sync(@c1)
// MOVADD-NEXT: aiex.cert.add(0, 1)
aie.device(npu2) {
  memref.global "private" constant @d1 : memref<2000xi32> = dense<0>
  aiex.cert.uc_dma_chain @c1 {
    aiex.cert.uc_dma_bd @d1, 0, 2000, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.mov(0, 16)
      aiex.cert.uc_dma_write_des_sync(@c1)
      aiex.cert.add(0, 1)
    }
  }
}

// -----

// Case 2: a uc_dma_write_des/wait_uc_dma pair, both on LOCAL register r0.
// Estimate = 32 + 8 (START_JOB) + 8 (uc_dma_write_des) + (16 + 8000 data) + 4
// (wait_uc_dma) + 4 (nop) + 4 (END_JOB) = 8076, same window as case 1. The
// wait handle must not be split away from its producer.
// This directive set is for the SECOND case: the concatenated stream already
// contains case 1's page/job(1) text before case 2's, so a bare, un-anchored
// page match here would otherwise land on case 1's page. Anchor past it
// first with a line unique to case 1 (its "mov(0, 16)"), then pin case 2's
// job contiguously with -NEXT.
// UCDMA: aiex.cert.mov(0, 16)
// UCDMA: aiex.cert.page
// UCDMA-NEXT: aiex.cert.job(1)
// UCDMA-NEXT: aiex.cert.uc_dma_write_des(0, @c1)
// UCDMA-NEXT: aiex.cert.wait_uc_dma(0)
// UCDMA-NEXT: aiex.cert.nop
aie.device(npu2) {
  memref.global "private" constant @d1 : memref<2000xi32> = dense<0>
  aiex.cert.uc_dma_chain @c1 {
    aiex.cert.uc_dma_bd @d1, 0, 2000, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des(0, @c1)
      aiex.cert.wait_uc_dma(0)
      aiex.cert.nop
    }
  }
}

// -----

// Case 3: the control case. Identical shape but on GLOBAL register r8, which
// survives a split, so the page DOES split exactly where the cost model
// predicts. Estimate = 32 + 8 (START_JOB) + 8 (mov) + 4 (uc_dma_write_des_sync)
// + (16 + 5200 data for @cA) + 8 (add) + 4 (uc_dma_write_des_sync) + (16 +
// 5200 data for @cB) + 4 (END_JOB) = 10500. Running cost first reaches the
// 4000 split_target right after uc_dma_write_des_sync(@cA) (5268), so the cut
// lands before `add`.
// This directive set is for the THIRD case: anchor past cases 1 and 2 first
// with a line unique to case 2 ("wait_uc_dma(0)"), then pin the two
// resulting pages contiguously with -NEXT (including the intervening
// job/page-closing braces) to prove the cut falls exactly between
// uc_dma_write_des_sync(@cA) and add(8, 1), as the cost model predicts.
// GLOBAL: aiex.cert.wait_uc_dma(0)
// GLOBAL: aiex.cert.page
// GLOBAL-NEXT: aiex.cert.job(1)
// GLOBAL-NEXT: aiex.cert.mov(8, 16)
// GLOBAL-NEXT: aiex.cert.uc_dma_write_des_sync(@cA)
// GLOBAL-NEXT: }
// GLOBAL-NEXT: }
// GLOBAL-NEXT: aiex.cert.page
// GLOBAL-NEXT: aiex.cert.job(2)
// GLOBAL-NEXT: aiex.cert.add(8, 1)
// GLOBAL-NEXT: aiex.cert.uc_dma_write_des_sync(@cB)
aie.device(npu2) {
  memref.global "private" constant @dA : memref<1300xi32> = dense<0>
  memref.global "private" constant @dB : memref<1300xi32> = dense<0>
  aiex.cert.uc_dma_chain @cA {
    aiex.cert.uc_dma_bd @dA, 0, 1300, false
  }
  aiex.cert.uc_dma_chain @cB {
    aiex.cert.uc_dma_bd @dB, 0, 1300, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.mov(8, 16)
      aiex.cert.uc_dma_write_des_sync(@cA)
      aiex.cert.add(8, 1)
      aiex.cert.uc_dma_write_des_sync(@cB)
    }
  }
}

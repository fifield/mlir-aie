//===- isolate_full_page_local_regs.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Controls for the local-register rule that isolate_full_page_local_regs_diag
// exercises. That rule tests values that actually CROSS the full-page op, not
// any local-register use whatsoever -- the blunt form would reject working
// designs. These two cases must therefore still isolate normally.

// RUN: aie-opt -cert-legalize-pages --split-input-file %s | FileCheck %s --check-prefix=ONESIDE
// RUN: aie-opt -cert-legalize-pages --split-input-file %s | FileCheck %s --check-prefix=GLOBAL

// Case 1: r0's live range lies entirely ahead of the cert.load_pdi, so nothing
// crosses. Isolation proceeds: the mov/write32_d pair stays together in the
// leading job, load_pdi takes a job of its own, and the trailing write32 gets
// a third. -NEXT pins the pair as adjacent inside one job.
// ONESIDE: aiex.cert.job(1)
// ONESIDE-NEXT: aiex.cert.mov(0, 68157440)
// ONESIDE-NEXT: aiex.cert.write32_d(0, 7) {address_is_reg}
// ONESIDE: aiex.cert.load_pdi(1, @config)
// ONESIDE: aiex.cert.write32(12288, 1)
module {
  aie.device(npu2) {
    aiex.cert.section @config {
      aiex.cert.page { aiex.cert.job(9) { aiex.cert.write32(0x2000, 20) } }
    }
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.mov(0, 68157440)
        aiex.cert.write32_d(0, 7) {address_is_reg}
        aiex.cert.load_pdi(1, @config)
        aiex.cert.write32(0x3000, 1)
      }
    }
  }
}

// -----

// Case 2: the identical crossing on a GLOBAL register r8. Globals are "shared
// among all the jobs" (isa-spec.yaml), so the value survives becoming distinct
// jobs and the rewrite must go through. FileCheck sees both cases' output in
// one stream (--split-input-file), so this prefix's first match is anchored on
// the r8 mov, which case 1 does not contain.
// GLOBAL: aiex.cert.mov(8, 68157440)
// GLOBAL: aiex.cert.load_pdi(1, @config)
// GLOBAL: aiex.cert.write32_d(8, 7) {address_is_reg}
module {
  aie.device(npu2) {
    aiex.cert.section @config {
      aiex.cert.page { aiex.cert.job(9) { aiex.cert.write32(0x2000, 20) } }
    }
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.mov(8, 68157440)
        aiex.cert.load_pdi(1, @config)
        aiex.cert.write32_d(8, 7) {address_is_reg}
      }
    }
  }
}

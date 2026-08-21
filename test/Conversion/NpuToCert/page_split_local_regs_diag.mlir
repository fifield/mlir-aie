//===- page_split_local_regs_diag.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// An oversized page that touches job-private local registers r0..r7 must
// never be silently split (its local-register state would not survive
// becoming two distinct jobs) and must never be silently emitted oversized
// either: it must produce a loud, specific diagnostic.
// The second case here is the control: the identical shape on global
// registers r8..r23 splits cleanly and must NOT produce any diagnostic.

// RUN: aie-opt -cert-legalize-pages --split-input-file --verify-diagnostics %s

// Case 1: LOCAL registers (r0). Estimate = 32 (header) + 8 (START_JOB) +
// 8 (mov) + 4 (uc_dma_write_des_sync) + (16 + 1300*4 = 5216 data for @c1) +
// 8 (add) + 4 (uc_dma_write_des_sync) + (16 + 1300*4 = 5216 data for @c2) +
// 4 (END_JOB) = 10500 bytes: over the 8192-byte hard limit. Every candidate
// cut falls inside job 1, which uses local registers, so none is legal; the
// page cannot be split and must error rather than emit an oversized page.
module {
  aie.device(npu2) {
    memref.global "private" constant @d1 : memref<1300xi32> = dense<0>
    memref.global "private" constant @d2 : memref<1300xi32> = dense<0>
    aiex.cert.uc_dma_chain @c1 {
      aiex.cert.uc_dma_bd @d1, 0, 1300, false
    }
    aiex.cert.uc_dma_chain @c2 {
      aiex.cert.uc_dma_bd @d2, 0, 1300, false
    }
    // expected-error@+1 {{cannot be split: job 1 uses job-private local registers r0..r7}}
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.mov(0, 16)
        aiex.cert.uc_dma_write_des_sync(@c1)
        aiex.cert.add(0, 1)
        aiex.cert.uc_dma_write_des_sync(@c2)
      }
    }
  }
}

// -----

// Case 2: the control. Identical shape and identical estimated cost (10500
// bytes), but on GLOBAL registers (r8), which survive a split. The page
// splits cleanly into two legal pages and emits NO diagnostic at all --
// --verify-diagnostics failing on an unexpected diagnostic is the assertion.
module {
  aie.device(npu2) {
    memref.global "private" constant @d1 : memref<1300xi32> = dense<0>
    memref.global "private" constant @d2 : memref<1300xi32> = dense<0>
    aiex.cert.uc_dma_chain @c1 {
      aiex.cert.uc_dma_bd @d1, 0, 1300, false
    }
    aiex.cert.uc_dma_chain @c2 {
      aiex.cert.uc_dma_bd @d2, 0, 1300, false
    }
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.mov(8, 16)
        aiex.cert.uc_dma_write_des_sync(@c1)
        aiex.cert.add(8, 1)
        aiex.cert.uc_dma_write_des_sync(@c2)
      }
    }
  }
}

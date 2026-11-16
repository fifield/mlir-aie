//===- cert_uc_dma_async_range.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// uC DMA wait handles are job-local register indices r0..r7.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(xcve3858) {
  aiex.cert.page {
    aiex.cert.job(1) {
      // expected-error@+1 {{attribute 'wait_handle' failed to satisfy constraint}}
      aiex.cert.wait_uc_dma(8)
    }
  }
}

// -----

aie.device(xcve3858) {
  aiex.cert.page {
    aiex.cert.job(1) {
      // expected-error@+1 {{attribute 'wait_handle' failed to satisfy constraint}}
      aiex.cert.uc_dma_write_des(-1, @chain)
    }
  }
}

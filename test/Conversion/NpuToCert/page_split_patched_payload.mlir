//===- page_split_patched_payload.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Keep payload patches and their uC-DMA pushes on the same page, since each
// page has a separate payload copy.

// RUN: aie-opt -cert-legalize-pages --split-input-file %s | FileCheck %s

// Interior cut: reject the greedy cut before the second write32, inside the
// @pB patch/push span. Split before the patch at cost 3988 (target 4000).
//
// CHECK: aiex.cert.page
// CHECK: aiex.cert.apply_offset_57(@pA
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainA)
// CHECK-NOT: aiex.cert.apply_offset_57(@pB
// CHECK: aiex.cert.page
// CHECK: aiex.cert.apply_offset_57(@pB
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainB)
aie.device(npu2) {
  memref.global "private" constant @pA : memref<980xi32> = dense<0>
  memref.global "private" constant @pB : memref<1000xi32> = dense<0>
  aiex.cert.uc_dma_chain @chainA {
    aiex.cert.uc_dma_bd @pA, 0, 980, false
  }
  aiex.cert.uc_dma_chain @chainB {
    aiex.cert.uc_dma_bd @pB, 0, 1000, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.apply_offset_57(@pA, 1, 0)
      aiex.cert.uc_dma_write_des_sync(@chainA)
      aiex.cert.apply_offset_57(@pB, 1, 0)
      aiex.cert.write32(0x1000, 1)
      aiex.cert.write32(0x1004, 2)
      aiex.cert.write32(0x1008, 3)
      aiex.cert.uc_dma_write_des_sync(@chainB)
    }
  }
}

// -----

// Job-boundary cut with an asynchronous push: r3 prevents splitting job 3.
// Reject the boundary before job 3 (cost 4000), which separates @qB's patch
// and push. Split before job 2 (cost 3980).
//
// CHECK: aiex.cert.page
// CHECK: aiex.cert.apply_offset_57(@qA
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainQA)
// CHECK-NOT: aiex.cert.apply_offset_57(@qB
// CHECK: aiex.cert.page
// CHECK: aiex.cert.apply_offset_57(@qB
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des(3, @chainQB)
aie.device(npu2) {
  memref.global "private" constant @qA : memref<975xi32> = dense<0>
  memref.global "private" constant @qB : memref<1000xi32> = dense<0>
  aiex.cert.uc_dma_chain @chainQA {
    aiex.cert.uc_dma_bd @qA, 0, 975, false
  }
  aiex.cert.uc_dma_chain @chainQB {
    aiex.cert.uc_dma_bd @qB, 0, 1000, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.apply_offset_57(@qA, 1, 0)
      aiex.cert.uc_dma_write_des_sync(@chainQA)
    }
    aiex.cert.job(2) {
      aiex.cert.apply_offset_57(@qB, 1, 0)
    }
    aiex.cert.job(3) {
      aiex.cert.uc_dma_write_des(3, @chainQB)
      aiex.cert.wait_uc_dma(3)
    }
  }
}

// -----

// An unpatched payload imposes no span constraint. Keep the greedy cut
// between the first and second write32 ops, before the push of @rB.
//
// CHECK: aiex.cert.page
// CHECK: aiex.cert.apply_offset_57(@rA
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainRA)
// CHECK: aiex.cert.write32(4096, 1)
// CHECK: aiex.cert.page
// CHECK: aiex.cert.write32(4100, 2)
// CHECK: aiex.cert.write32(4104, 3)
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainRB)
aie.device(npu2) {
  memref.global "private" constant @rA : memref<980xi32> = dense<0>
  memref.global "private" constant @rB : memref<1000xi32> = dense<0>
  aiex.cert.uc_dma_chain @chainRA {
    aiex.cert.uc_dma_bd @rA, 0, 980, false
  }
  aiex.cert.uc_dma_chain @chainRB {
    aiex.cert.uc_dma_bd @rB, 0, 1000, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.apply_offset_57(@rA, 1, 0)
      aiex.cert.uc_dma_write_des_sync(@chainRA)
      aiex.cert.write32(0x1000, 1)
      aiex.cert.write32(0x1004, 2)
      aiex.cert.write32(0x1008, 3)
      aiex.cert.uc_dma_write_des_sync(@chainRB)
    }
  }
}

// -----

// Repeated pushes need the same patched copy. Keep this page intact: its cost
// exceeds the 8000-byte split trigger but fits the 8192-byte hardware limit.
//
// CHECK-LABEL: aie.device(npu2) @repeated_push
// CHECK: aiex.cert.page
// CHECK: aiex.cert.apply_offset_57(@sA
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainSA)
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainSB)
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainSA)
// CHECK-NOT: aiex.cert.page
aie.device(npu2) @repeated_push {
  memref.global "private" constant @sA : memref<980xi32> = dense<0>
  memref.global "private" constant @sB : memref<1000xi32> = dense<0>
  aiex.cert.uc_dma_chain @chainSA {
    aiex.cert.uc_dma_bd @sA, 0, 980, false
  }
  aiex.cert.uc_dma_chain @chainSB {
    aiex.cert.uc_dma_bd @sB, 0, 1000, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.apply_offset_57(@sA, 1, 0)
      aiex.cert.uc_dma_write_des_sync(@chainSA)
      aiex.cert.write32(0x1000, 1)
      aiex.cert.write32(0x1004, 2)
      aiex.cert.write32(0x1008, 3)
      aiex.cert.uc_dma_write_des_sync(@chainSB)
      aiex.cert.uc_dma_write_des_sync(@chainSA)
    }
  }
}

// -----

// Patches accumulate across pushes, including through distinct chains sharing
// a payload. Keep both patches with the asynchronous push; its global wait
// handle allows the wait to move to a later page.
//
// CHECK-LABEL: aie.device(npu2) @shared_patched_payload
// CHECK: aiex.cert.page
// CHECK: aiex.cert.apply_offset_57(@tA, 1, 0)
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainTA)
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainTB)
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.apply_offset_57(@tA, 1, 2)
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des(8, @chainTA2)
// CHECK: aiex.cert.page
// CHECK: aiex.cert.wait_uc_dma(8)
// CHECK-NOT: aiex.cert.page
aie.device(npu2) @shared_patched_payload {
  memref.global "private" constant @tA : memref<980xi32> = dense<0>
  memref.global "private" constant @tB : memref<1000xi32> = dense<0>
  aiex.cert.uc_dma_chain @chainTA {
    aiex.cert.uc_dma_bd @tA, 0, 980, false
  }
  aiex.cert.uc_dma_chain @chainTB {
    aiex.cert.uc_dma_bd @tB, 0, 1000, false
  }
  aiex.cert.uc_dma_chain @chainTA2 {
    aiex.cert.uc_dma_bd @tA, 4096, 980, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.apply_offset_57(@tA, 1, 0)
      aiex.cert.uc_dma_write_des_sync(@chainTA)
      aiex.cert.write32(0x1000, 1)
      aiex.cert.write32(0x1004, 2)
      aiex.cert.write32(0x1008, 3)
      aiex.cert.uc_dma_write_des_sync(@chainTB)
      aiex.cert.apply_offset_57(@tA, 1, 2)
      aiex.cert.uc_dma_write_des(8, @chainTA2)
      aiex.cert.wait_uc_dma(8)
    }
  }
}

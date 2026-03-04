//===- cert_page_emission.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s
// RUN: aie-translate %s -aie-cert-to-asm | FileCheck %s --check-prefix=ASM

module {
  aie.device(npu2) {
    // Test 1: Single page with one job
    // CHECK: aiex.cert.page
    aiex.cert.page {
      // CHECK: aiex.cert.job(1)
      aiex.cert.job(1) {
        // CHECK: aiex.cert.write32(4096, 42)
        aiex.cert.write32(0x1000, 42)
      }
    }

    // Test 2: Page within attach_to_group
    // CHECK: aiex.cert.attach_to_group(0)
    aiex.cert.attach_to_group(0) {
      // CHECK: aiex.cert.page
      aiex.cert.page {
        // CHECK: aiex.cert.job(2)
        aiex.cert.job(2) {
          // CHECK: aiex.cert.write32(8192, 100)
          aiex.cert.write32(0x2000, 100)
          // CHECK: aiex.cert.nop
          aiex.cert.nop
        }
      }
    }
    aiex.cert.attach_to_group(1) {
      // CHECK: aiex.cert.page
      aiex.cert.page {
        // CHECK: aiex.cert.job(3)
        aiex.cert.job(3) {
          // CHECK: aiex.cert.maskwrite32(12288, 200, 255)
          aiex.cert.maskwrite32(0x3000, 200, 0xFF)
        }
      }
    }
  }
}

// ASM-LABEL: ; Code
// ASM: .attach_to_group 0

// ASM: .align 16
// ASM: START_JOB 1
// ASM:   WRITE_32               0x00001000, 0x0000002a
// ASM: END_JOB
// ASM: .eop

// ASM: .align 16
// ASM: START_JOB 2
// ASM:   WRITE_32               0x00002000, 0x00000064
// ASM:   NOP
// ASM: END_JOB
// ASM: .eop
// ASM: EOF

// ASM: .attach_to_group 1
// ASM: .align 16
// ASM: START_JOB 3
// ASM:   MASK_WRITE_32          0x00003000, 0x000000c8, 0x000000ff
// ASM: END_JOB
// ASM: .eop
// ASM: EOF

//===- cert_to_asm_sections.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate -aie-cert-to-asm %s | FileCheck %s

// Test that cert.page, cert.section, and cert.load_pdi emit correct assembly

// CHECK: config_device:
// CHECK: .include config_device.asm
// CHECK: .endl config_device

// CHECK: .align 16
// CHECK: START_JOB 1
// CHECK:   LOAD_PDI               1, @config_device
// CHECK: END_JOB

// CHECK: .eop

// CHECK: .align 16
// CHECK: START_JOB 2
// CHECK:   WRITE_32               0x00001000, 0x00000064
// CHECK: END_JOB

// CHECK: .eop

// CHECK: EOF

module {
  aie.device(npu3z) {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.load_pdi(1, @config_device)
      }
    }
    aiex.cert.page {
      aiex.cert.job(2) {
        aiex.cert.write32(0x1000, 100)
      }
    }
    aiex.cert.section @config_device {
      aiex.cert.page {
        aiex.cert.job(100) {
          aiex.cert.write32(0x2000, 123)
        }
      }
    }
  }
}

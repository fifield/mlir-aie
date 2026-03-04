//===- cert_load_pdi_emission.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s
// RUN: rm -f device_config.asm pdi_layer1.asm pdi_layer2.asm
// RUN: aie-translate %s -aie-cert-to-asm -o %t.asm
// RUN: FileCheck %s --check-prefix=ASM < %t.asm
// RUN: FileCheck %s --check-prefix=DEVICE_CONFIG < device_config.asm
// RUN: FileCheck %s --check-prefix=PDI_LAYER1 < pdi_layer1.asm
// RUN: FileCheck %s --check-prefix=PDI_LAYER2 < pdi_layer2.asm

module {
  aie.device(npu2) {
    // Test 1: Define a configuration section
    // CHECK: aiex.cert.section @device_config
    aiex.cert.section @device_config {
      // CHECK: aiex.cert.page
      aiex.cert.page {
        // CHECK: aiex.cert.job(100)
        aiex.cert.job(100) {
          // CHECK: aiex.cert.write32(1703936, 1)
          aiex.cert.write32(0x1A0000, 0x1)
          // CHECK: aiex.cert.write32(1703940, 2)
          aiex.cert.write32(0x1A0004, 0x2)
        }
      }
    }

    // Test 2: Multiple PDI sections
    // CHECK: aiex.cert.section @pdi_layer1
    aiex.cert.section @pdi_layer1 {
      // CHECK: aiex.cert.page
      aiex.cert.page {
        // CHECK: aiex.cert.job(101)
        aiex.cert.job(101) {
          // CHECK: aiex.cert.write32(8192, 10)
          aiex.cert.write32(0x2000, 10)
        }
      }
    }

    // CHECK: aiex.cert.section @pdi_layer2
    aiex.cert.section @pdi_layer2 {
      // CHECK: aiex.cert.page
      aiex.cert.page {
        // CHECK: aiex.cert.job(102)
        aiex.cert.job(102) {
          // CHECK: aiex.cert.write32(8192, 20)
          aiex.cert.write32(0x2000, 20)
        }
      }
    }

    // Test 3: Main control code that loads PDIs
    // CHECK: aiex.cert.attach_to_group(0)
    aiex.cert.attach_to_group(0) {
      // CHECK: aiex.cert.page
      aiex.cert.page {
        // Load device configuration
        // CHECK: aiex.cert.job(1)
        aiex.cert.job(1) {
          // CHECK: aiex.cert.load_pdi(1, @device_config)
          aiex.cert.load_pdi(1, @device_config)
        }
      }

      // CHECK: aiex.cert.page
      aiex.cert.page {
        // Do some work after config
        // CHECK: aiex.cert.job(2)
        aiex.cert.job(2) {
          // CHECK: aiex.cert.write32(4096, 42)
          aiex.cert.write32(0x1000, 42)
        }
      }

      // CHECK: aiex.cert.page
      aiex.cert.page {
        // Load layer 1 PDI
        // CHECK: aiex.cert.job(3)
        aiex.cert.job(3) {
          // CHECK: aiex.cert.load_pdi(2, @pdi_layer1)
          aiex.cert.load_pdi(2, @pdi_layer1)
        }
      }

      // CHECK: aiex.cert.page
      aiex.cert.page {
        // Load layer 2 PDI
        // CHECK: aiex.cert.job(4)
        aiex.cert.job(4) {
          // CHECK: aiex.cert.load_pdi(3, @pdi_layer2)
          aiex.cert.load_pdi(3, @pdi_layer2)
        }
      }
    }
  }
}

// Assembly tests

// ASM-LABEL: ; Code

// ASM: device_config:
// ASM: .include device_config.asm
// ASM: .endl device_config

// ASM: pdi_layer1:
// ASM: .include pdi_layer1.asm
// ASM: .endl pdi_layer1

// ASM: pdi_layer2:
// ASM: .include pdi_layer2.asm
// ASM: .endl pdi_layer2

// ASM: .align 16
// ASM: START_JOB 1
// ASM:   LOAD_PDI               1, @device_config
// ASM: END_JOB
// ASM: .eop

// ASM: .align 16
// ASM: START_JOB 2
// ASM:   WRITE_32               0x00001000, 0x0000002a
// ASM: END_JOB
// ASM: .eop

// ASM: .align 16
// ASM: START_JOB 3
// ASM:   LOAD_PDI               2, @pdi_layer1
// ASM: END_JOB
// ASM: .eop

// ASM: .align 16
// ASM: START_JOB 4
// ASM:   LOAD_PDI               3, @pdi_layer2
// ASM: END_JOB
// ASM: .eop

// ASM: EOF

// Section file tests

// DEVICE_CONFIG: ;
// DEVICE_CONFIG: ;text
// DEVICE_CONFIG: ;
// DEVICE_CONFIG: .align 16
// DEVICE_CONFIG: START_JOB 100
// DEVICE_CONFIG:   WRITE_32               0x001a0000, 0x00000001
// DEVICE_CONFIG:   WRITE_32               0x001a0004, 0x00000002
// DEVICE_CONFIG: END_JOB
// DEVICE_CONFIG: .eop
// DEVICE_CONFIG: EOF

// PDI_LAYER1: ;
// PDI_LAYER1: ;text
// PDI_LAYER1: ;
// PDI_LAYER1: .align 16
// PDI_LAYER1: START_JOB 101
// PDI_LAYER1:   WRITE_32               0x00002000, 0x0000000a
// PDI_LAYER1: END_JOB
// PDI_LAYER1: .eop
// PDI_LAYER1: EOF

// PDI_LAYER2: ;
// PDI_LAYER2: ;text
// PDI_LAYER2: ;
// PDI_LAYER2: .align 16
// PDI_LAYER2: START_JOB 102
// PDI_LAYER2:   WRITE_32               0x00002000, 0x00000014
// PDI_LAYER2: END_JOB
// PDI_LAYER2: .eop
// PDI_LAYER2: EOF

//===- cert_section_emission.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s
// RUN: rm -f config_a.asm save_state.asm restore_state.asm
// RUN: aie-translate %s -aie-cert-to-asm -o %t.asm
// RUN: FileCheck %s --check-prefix=ASM < %t.asm
// RUN: FileCheck %s --check-prefix=CONFIG_A < config_a.asm
// RUN: FileCheck %s --check-prefix=SAVE_STATE < save_state.asm
// RUN: FileCheck %s --check-prefix=RESTORE_STATE < restore_state.asm

module {
  aie.device(npu2) {
    // Test 1: Simple section with one page and one job
    // CHECK: aiex.cert.section @config_a
    aiex.cert.section @config_a {
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

    // Test 2: Multiple sections
    // CHECK: aiex.cert.section @save_state
    aiex.cert.section @save_state {
      // CHECK: aiex.cert.page
      aiex.cert.page {
        // CHECK: aiex.cert.job(200)
        aiex.cert.job(200) {
          // CHECK: aiex.cert.write32(8192, 0)
          aiex.cert.write32(0x2000, 0x0)
        }
      }
    }

    // CHECK: aiex.cert.section @restore_state
    aiex.cert.section @restore_state {
      // CHECK: aiex.cert.page
      aiex.cert.page {
        // CHECK: aiex.cert.job(201)
        aiex.cert.job(201) {
          // CHECK: aiex.cert.write32(8192, 1)
          aiex.cert.write32(0x2000, 0x1)
        }
      }
    }

    // Test 3: Main control code (to verify sections are emitted before main code)
    // CHECK: aiex.cert.attach_to_group(0)
    aiex.cert.attach_to_group(0) {
      // CHECK: aiex.cert.page
      aiex.cert.page {
        // CHECK: aiex.cert.job(1)
        aiex.cert.job(1) {
          // CHECK: aiex.cert.write32(4096, 42)
          aiex.cert.write32(0x1000, 42)
        }
      }
    }
  }
}

// Assembly tests

// ASM-LABEL: ; Code

// ASM: config_a:
// ASM: .include config_a.asm
// ASM: .endl config_a

// ASM: save_state:
// ASM: .include save_state.asm
// ASM: .endl save_state

// ASM: restore_state:
// ASM: .include restore_state.asm
// ASM: .endl restore_state

// ASM: .align 16
// ASM: START_JOB 1
// ASM:   WRITE_32               0x00001000, 0x0000002a
// ASM: END_JOB
// ASM: .eop
// ASM: EOF

// Section file tests

// CONFIG_A: ;
// CONFIG_A: ;text
// CONFIG_A: ;
// CONFIG_A: .align 16
// CONFIG_A: START_JOB 100
// CONFIG_A:   WRITE_32               0x001a0000, 0x00000001
// CONFIG_A:   WRITE_32               0x001a0004, 0x00000002
// CONFIG_A: END_JOB
// CONFIG_A: .eop
// CONFIG_A: EOF

// SAVE_STATE: ;
// SAVE_STATE: ;text
// SAVE_STATE: ;
// SAVE_STATE: .align 16
// SAVE_STATE: START_JOB 200
// SAVE_STATE:   WRITE_32               0x00002000, 0x00000000
// SAVE_STATE: END_JOB
// SAVE_STATE: .eop
// SAVE_STATE: EOF

// RESTORE_STATE: ;
// RESTORE_STATE: ;text
// RESTORE_STATE: ;
// RESTORE_STATE: .align 16
// RESTORE_STATE: START_JOB 201
// RESTORE_STATE:   WRITE_32               0x00002000, 0x00000001
// RESTORE_STATE: END_JOB
// RESTORE_STATE: .eop
// RESTORE_STATE: EOF

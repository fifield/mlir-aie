//===- cert_preempt_emission.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s
// RUN: rm -f save_context.asm restore_context.asm save_layer1.asm restore_layer1.asm
// RUN: aie-translate %s -aie-cert-to-asm -o %t.asm
// RUN: FileCheck %s --check-prefix=ASM < %t.asm
// RUN: FileCheck %s --check-prefix=SAVE_CONTEXT < save_context.asm
// RUN: FileCheck %s --check-prefix=RESTORE_CONTEXT < restore_context.asm
// RUN: FileCheck %s --check-prefix=SAVE_LAYER1 < save_layer1.asm
// RUN: FileCheck %s --check-prefix=RESTORE_LAYER1 < restore_layer1.asm

// NOTE: Sections are emitted as separate .asm files referenced via .include
// directives to match the format required by aiebu-asm.

module {
  aie.device(npu2) {
    // Test 1: Define save and restore sections for preemption
    // CHECK: aiex.cert.section @save_context
    aiex.cert.section @save_context {
      // CHECK: aiex.cert.page
      aiex.cert.page {
        // CHECK: aiex.cert.job(100)
        aiex.cert.job(100) {
          // Save tile state
          // CHECK: aiex.cert.write32(35258368, 0)
          aiex.cert.write32(0x21A0000, 0x0)
        }
      }
    }

    // CHECK: aiex.cert.section @restore_context
    aiex.cert.section @restore_context {
      // CHECK: aiex.cert.page
      aiex.cert.page {
        // CHECK: aiex.cert.job(101)
        aiex.cert.job(101) {
          // Restore tile state
          // CHECK: aiex.cert.write32(35258368, 1)
          aiex.cert.write32(0x21A0000, 0x1)
        }
      }
    }

    // Multiple save/restore pairs
    // CHECK: aiex.cert.section @save_layer1
    aiex.cert.section @save_layer1 {
      // CHECK: aiex.cert.page
      aiex.cert.page {
        // CHECK: aiex.cert.job(102)
        aiex.cert.job(102) {
          // CHECK: aiex.cert.write32(8192, 10)
          aiex.cert.write32(0x2000, 10)
        }
      }
    }

    // CHECK: aiex.cert.section @restore_layer1
    aiex.cert.section @restore_layer1 {
      // CHECK: aiex.cert.page
      aiex.cert.page {
        // CHECK: aiex.cert.job(103)
        aiex.cert.job(103) {
          // CHECK: aiex.cert.write32(8192, 11)
          aiex.cert.write32(0x2000, 11)
        }
      }
    }

    // Test 2: Main control code with preemption points
    // CHECK: aiex.cert.attach_to_group(0)
    aiex.cert.attach_to_group(0) {
      // CHECK: aiex.cert.page
      aiex.cert.page {
        // Initial work
        // CHECK: aiex.cert.job(1)
        aiex.cert.job(1) {
          // CHECK: aiex.cert.write32(4096, 42)
          aiex.cert.write32(0x1000, 42)
        }
      }

      // CHECK: aiex.cert.page
      aiex.cert.page {
        // First preemption point
        // CHECK: aiex.cert.job(2)
        aiex.cert.job(2) {
          // CHECK: aiex.cert.preempt(0, @save_context, @restore_context)
          aiex.cert.preempt(0, @save_context, @restore_context)
        }
      }

      // CHECK: aiex.cert.page
      aiex.cert.page {
        // More work
        // CHECK: aiex.cert.job(3)
        aiex.cert.job(3) {
          // CHECK: aiex.cert.write32(4096, 100)
          aiex.cert.write32(0x1000, 100)
        }
      }

      // CHECK: aiex.cert.page
      aiex.cert.page {
        // Second preemption point with different ID and sections
        // CHECK: aiex.cert.job(4)
        aiex.cert.job(4) {
          // CHECK: aiex.cert.preempt(1, @save_layer1, @restore_layer1)
          aiex.cert.preempt(1, @save_layer1, @restore_layer1)
        }
      }
    }
  }
}

// Assembly tests

// ASM-LABEL: ; Code

// ASM: .attach_to_group 0

// ASM: save_context:
// ASM: .include save_context.asm
// ASM: .endl save_context

// ASM: restore_context:
// ASM: .include restore_context.asm
// ASM: .endl restore_context

// ASM: save_layer1:
// ASM: .include save_layer1.asm
// ASM: .endl save_layer1

// ASM: restore_layer1:
// ASM: .include restore_layer1.asm
// ASM: .endl restore_layer1

// ASM: .align 16
// ASM: START_JOB 1
// ASM:   WRITE_32               0x00001000, 0x0000002a
// ASM: END_JOB
// ASM: .eop

// ASM: .align 16
// ASM: START_JOB 2
// ASM:   PREEMPT                0x0000, @save_context, @restore_context
// ASM: END_JOB
// ASM: .eop

// ASM: .align 16
// ASM: START_JOB 3
// ASM:   WRITE_32               0x00001000, 0x00000064
// ASM: END_JOB
// ASM: .eop

// ASM: .align 16
// ASM: START_JOB 4
// ASM:   PREEMPT                0x0001, @save_layer1, @restore_layer1
// ASM: END_JOB
// ASM: .eop

// ASM: EOF

// Section file tests

// SAVE_CONTEXT: ;
// SAVE_CONTEXT: ;text
// SAVE_CONTEXT: ;
// SAVE_CONTEXT: .align 16
// SAVE_CONTEXT: START_JOB 100
// SAVE_CONTEXT:   WRITE_32               0x021a0000, 0x00000000
// SAVE_CONTEXT: END_JOB
// SAVE_CONTEXT: .eop
// SAVE_CONTEXT: EOF

// RESTORE_CONTEXT: ;
// RESTORE_CONTEXT: ;text
// RESTORE_CONTEXT: ;
// RESTORE_CONTEXT: .align 16
// RESTORE_CONTEXT: START_JOB 101
// RESTORE_CONTEXT:   WRITE_32               0x021a0000, 0x00000001
// RESTORE_CONTEXT: END_JOB
// RESTORE_CONTEXT: .eop
// RESTORE_CONTEXT: EOF

// SAVE_LAYER1: ;
// SAVE_LAYER1: ;text
// SAVE_LAYER1: ;
// SAVE_LAYER1: .align 16
// SAVE_LAYER1: START_JOB 102
// SAVE_LAYER1:   WRITE_32               0x00002000, 0x0000000a
// SAVE_LAYER1: END_JOB
// SAVE_LAYER1: .eop
// SAVE_LAYER1: EOF

// RESTORE_LAYER1: ;
// RESTORE_LAYER1: ;text
// RESTORE_LAYER1: ;
// RESTORE_LAYER1: .align 16
// RESTORE_LAYER1: START_JOB 103
// RESTORE_LAYER1:   WRITE_32               0x00002000, 0x0000000b
// RESTORE_LAYER1: END_JOB
// RESTORE_LAYER1: .eop
// RESTORE_LAYER1: EOF

//===- cert_to_asm_section_dir.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// A cert.section is emitted as two things: a `.include <section>.asm` line in
// the main assembly, and a separate file holding the body. The include names
// the file WITHOUT any directory component, so an assembler resolves it
// relative to the file that contains it -- the two must end up in the same
// directory.
//
// aiecc writes the main assembly into its work directory. Before the fix it
// gave the translation no path, so the sections went to the process's current
// directory instead, and `aiebu-asm` failed with "<section>.asm file not
// found" on every module with more than one device. Single-device modules emit
// no sections and so never noticed.
//
// This pins the contract aiecc depends on: given a directory, the sections go
// there. The default (no option) stays the historical current-directory
// behaviour, checked below too, so no existing caller changes.

// RUN: rm -rf %t.into %t.cwd && mkdir -p %t.into %t.cwd

// With a directory, the section lands there and NOT in the current directory.
// RUN: cd %t.cwd && aie-translate -aie-cert-to-asm --cert-section-dir=%t.into %s -o %t.into/main.asm
// The explicit ls makes a missing section file a FAILure rather than letting
// the redirect below turn it into an unresolved test.
// RUN: ls %t.into/config_device.asm
// RUN: FileCheck %s --check-prefix=MAIN < %t.into/main.asm
// RUN: FileCheck %s --check-prefix=BODY < %t.into/config_device.asm
// RUN: not ls %t.cwd/config_device.asm

// Without it, the historical behaviour: the section is written relative to the
// process's current directory.
// RUN: cd %t.cwd && aie-translate -aie-cert-to-asm %s -o %t.into/main2.asm
// RUN: FileCheck %s --check-prefix=BODY < %t.cwd/config_device.asm

// The include carries no directory component -- which is the whole reason the
// two files have to be siblings.
// MAIN: config_device:
// MAIN-NEXT: .include config_device.asm
// MAIN-NEXT: .endl config_device

// BODY: START_JOB 1
// BODY: WRITE_32

module {
  aie.device(npu3) {
    aiex.cert.section @config_device {
      aiex.cert.page {
        aiex.cert.job(1) {
          aiex.cert.write32(0x1A0000, 0x1)
        }
      }
    }
  }
}

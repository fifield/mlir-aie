//===- entry_configuration_section.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -aie-npu-to-cert -split-input-file %s | FileCheck %s

// The entry's configuration is delivered like any other device's: as a
// cert.section that a LOAD_PDI names. Job 1 ({cert.configure}) only loads it,
// so the firmware skips it while the entry is still loaded. A generated AIE4
// configuration starts with the reset of the partition.
// 0x11111111 = 286331153, 0x22222222 = 572662306.

// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.job(1) {
// CHECK-NEXT: aiex.cert.load_pdi({{[0-9]+}}, @main)
// CHECK-NEXT: } {cert.configure}
// CHECK-NEXT: aiex.cert.job({{[0-9]+}}) {
// CHECK-NEXT: aiex.cert.write32(8192, 572662306)
// CHECK-NEXT: }{{$}}
// CHECK: aiex.cert.section @main
// CHECK: aiex.cert.write32(4096, 286331153)
module {
  aie.device(npu3) @main {
    aie.runtime_sequence @configure() attributes {aie.generated_configure} {
      %a = arith.constant 4096 : i32
      %v = arith.constant 0x11111111 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
    aie.runtime_sequence @seq() {
      %a = arith.constant 8192 : i32
      %v = arith.constant 0x22222222 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }
}

// -----

// An explicit load of the entry at the head of the program (IRON's implicit
// full-ELF load) is the same load: there is one.

// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.job(1) {
// CHECK-NEXT: aiex.cert.load_pdi({{[0-9]+}}, @main)
// CHECK-NEXT: } {cert.configure}
// CHECK-NOT: aiex.cert.load_pdi
// CHECK: aiex.cert.section @main
module {
  aie.device(npu3) @main {
    aie.runtime_sequence @configure() attributes {aie.generated_configure} {
      %a = arith.constant 4096 : i32
      %v = arith.constant 0x11111111 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
    aie.runtime_sequence @seq() {
      aiex.npu.load_pdi {device_ref = @main}
      %a = arith.constant 8192 : i32
      %v = arith.constant 0x22222222 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }
}

// -----

// A program on more than one microcontroller keeps its configuration as job
// 1: a LOAD_PDI on uC 0 alone would leave the other uCs out of step.

// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.job(1) {
// CHECK: aiex.cert.write32(4096, 286331153)
// CHECK-NEXT: } {cert.configure}
// CHECK-NOT: aiex.cert.section @main
module {
  aie.device(npu3) @main {
    aie.runtime_sequence @configure() attributes {aie.generated_configure} {
      %a = arith.constant 4096 : i32
      %v = arith.constant 0x11111111 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
    aie.runtime_sequence @seq() {
      aiex.cert.attach_to_group(2) {
        aiex.cert.job(5) {
          aiex.cert.write32(4096, 1)
        }
      }
    }
  }
}

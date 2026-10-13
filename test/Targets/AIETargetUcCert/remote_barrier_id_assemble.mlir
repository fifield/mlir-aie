//===- remote_barrier_id_assemble.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aiebu encodes $rbN as firmware barrier N+1 (REMOTE_BARRIER is opcode 0x12,
// barrier id in byte 2), so rb0 and rb6 must encode as 0x01 and 0x07.

// REQUIRES: aiebu_asm
// RUN: rm -rf remote_barrier_id_assemble && mkdir -p remote_barrier_id_assemble
// RUN: aie-translate %s -aie-cert-to-asm -o remote_barrier_id_assemble/ctrl.asm
// RUN: aiebu-asm -t aie4 -c remote_barrier_id_assemble/ctrl.asm -o remote_barrier_id_assemble/ctrl.elf
// RUN: llvm-readelf -x .ctrltext.0.0 remote_barrier_id_assemble/ctrl.elf | FileCheck %s
// CHECK: 12000100 03000000
// CHECK: 12000700 03000000

module {
  aie.device(npu3) {
    aiex.cert.job(1) {
      aiex.cert.remote_barrier(rb0, 0x3)
      aiex.cert.remote_barrier(rb6, 0x3)
    }
  }
}

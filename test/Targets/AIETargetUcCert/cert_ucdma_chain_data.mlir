//===- cert_ucdma_chain_data.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Offline guard for uC-DMA descriptor + payload emission, the mechanism behind
// cert.uc_dma_write_des / cert.wait_uc_dma / cert.uc_dma_write_des_sync and
// (already, in every device-config lowering) aie.buffer initial values and
// device-config blockwrites.
//
// Two things are pinned here, both of which have silent failure modes on
// hardware:
//
//  1. UC_DMA_BD operand order and next_bd polarity. The emitted operands are
//     `0, <local_address>, @<remote_address>, <length>, 0, <next_bd>` -- note
//     that MLIR's `local_address` is the DESTINATION AIE-array address and
//     MLIR's `remote_address` is the symbol holding the SOURCE data, the
//     inverse of the ISA spec's prose. next_bd is "another BD follows", so a
//     chain must end in 0; emitting 1 there runs the DMA off the end of the
//     chain, and emitting 0 on a non-final BD silently drops every following
//     BD (the destination block just stays at whatever was there before).
//
//  2. The payload words. cert.uc_dma_bd's remote_address is a
//     FlatSymbolRefAttr, and the emitter (AIETargetUcCert.cpp,
//     emitUcDmaBdData) resolves it to a device-level `memref.global "private"
//     constant` and inlines that global's DenseIntElementsAttr as `.long`s
//     under a label of the same name, once per symbol even if several BDs
//     reference it. If the words, their order, or the label went wrong, the
//     assembler would still succeed and the hardware would just transfer the
//     wrong bytes.

// RUN: aie-translate -aie-cert-to-asm %s | FileCheck %s

module {
  aie.device(xcve3858) {
    memref.global "private" constant @ucdma_head : memref<4xi32> =
      dense<[0x11110001, 0x11110002, 0x11110003, 0x11110004]>
    memref.global "private" constant @ucdma_tail : memref<2xi32> =
      dense<[0x22220001, 0x22220002]>
    // Splat form: same DenseIntElementsAttr path, repeated words.
    memref.global "private" constant @ucdma_splat : memref<3xi32> = dense<0xA5A5A501>

    // Two-BD chain: the non-final BD carries next_bd = true (1), the final one
    // false (0). @ucdma_splat is referenced by two different chains to pin the
    // emit-once behaviour.
    aiex.cert.uc_dma_chain @chain_multi {
      aiex.cert.uc_dma_bd @ucdma_head, 0x600500, 4, true
      aiex.cert.uc_dma_bd @ucdma_tail, 0x600900, 2, false
    }
    aiex.cert.uc_dma_chain @chain_single {
      aiex.cert.uc_dma_bd @ucdma_splat, 0x600A00, 3, false
    }

    aiex.cert.job(7) {
      aiex.cert.uc_dma_write_des(0, @chain_multi)
      aiex.cert.wait_uc_dma(0)
      aiex.cert.uc_dma_write_des_sync(@chain_single)
    }
  }
}

// The async pair keeps its wait handle in a job-private register, and the sync
// form takes the chain symbol directly.
// CHECK-LABEL: START_JOB 7
// CHECK:         UC_DMA_WRITE_DES       $r0, @chain_multi
// CHECK-NEXT:    WAIT_UC_DMA            $r0
// CHECK-NEXT:    uC_DMA_WRITE_DES_SYNC  @chain_single
// CHECK:       END_JOB

// Chain bodies. Operand order is
//   UC_DMA_BD <col>, <dest AIE-array addr>, @<source symbol>, <words>, 0, <next_bd>
// CHECK-LABEL: chain_multi:
// CHECK-NEXT:    UC_DMA_BD       0, 0x00600500, @ucdma_head, 4, 0, 1
// CHECK-NEXT:    UC_DMA_BD       0, 0x00600900, @ucdma_tail, 2, 0, 0
// CHECK-LABEL: chain_single:
// CHECK-NEXT:    UC_DMA_BD       0, 0x00600a00, @ucdma_splat, 3, 0, 0

// Payload words, in order, under a label named after the global.
// CHECK-LABEL: ucdma_head:
// CHECK-NEXT:    .long           0x11110001
// CHECK-NEXT:    .long           0x11110002
// CHECK-NEXT:    .long           0x11110003
// CHECK-NEXT:    .long           0x11110004
// CHECK-LABEL: ucdma_tail:
// CHECK-NEXT:    .long           0x22220001
// CHECK-NEXT:    .long           0x22220002
// CHECK-LABEL: ucdma_splat:
// CHECK-NEXT:    .long           0xa5a5a501
// CHECK-NEXT:    .long           0xa5a5a501
// CHECK-NEXT:    .long           0xa5a5a501

// Nothing else follows: no duplicate emission of a payload, and no stray BD.
// CHECK-NOT:     UC_DMA_BD
// CHECK-NOT:     .long

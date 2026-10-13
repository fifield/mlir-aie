//===- uc_dma_chain_groups.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-cert-to-asm --split-input-file %s | FileCheck %s
// RUN: aie-opt --cert-legalize-pages --split-input-file %s | aie-translate --aie-cert-to-asm --split-input-file | FileCheck %s --check-prefix=OPT

// A device-level chain must be emitted in every uC block that uses it. The
// payload must be duplicated too because each block has its own label namespace.
module {
  aie.device(npu3) {
    memref.global "private" constant @shared_data : memref<1xi32> = dense<[1]>
    aiex.cert.uc_dma_chain @shared {
      aiex.cert.uc_dma_bd @shared_data, 4321, 1, false
    }
    aiex.cert.uc_dma_chain @other {
      aiex.cert.uc_dma_bd @shared_data, 4322, 1, false
    }
    aiex.cert.attach_to_group(1) {
      aiex.cert.job(0) {
        aiex.cert.uc_dma_write_des(0, @shared)
        aiex.cert.wait_uc_dma(0)
      }
    }
    // Separate blocks for the same uC share one emitted chain and payload.
    aiex.cert.attach_to_group(1) {
      aiex.cert.job(2) {
        aiex.cert.uc_dma_write_des_sync(@shared)
        aiex.cert.uc_dma_write_des_sync(@other)
      }
    }
    aiex.cert.attach_to_group(2) {
      aiex.cert.job(1) {
        aiex.cert.uc_dma_write_des_sync(@shared)
      }
    }
  }
}

// CHECK-LABEL: .attach_to_group 0
// CHECK:       EOF
// CHECK:       ; Data (chains)
// CHECK-NOT:   shared:
// CHECK:       ; Data
// CHECK-NOT:   shared_data:
// CHECK-LABEL: .attach_to_group 1
// CHECK:         UC_DMA_WRITE_DES       $r0, @shared
// CHECK-NEXT:    WAIT_UC_DMA            $r0
// CHECK:       EOF
// CHECK:       ; Data (chains)
// CHECK:       shared:
// CHECK-NEXT:    UC_DMA_BD       0, 0x000010e1, @shared_data, 1, 0, 0
// CHECK-NOT:   shared:
// CHECK-NOT:   shared_data:
// CHECK:       other:
// CHECK-NEXT:    UC_DMA_BD       0, 0x000010e2, @shared_data, 1, 0, 0
// CHECK:       shared_data:
// CHECK-NEXT:    .long           0x00000001
// CHECK-NOT:   shared_data:
// CHECK-LABEL: .attach_to_group 2
// CHECK:         uC_DMA_WRITE_DES_SYNC  @shared
// CHECK:       EOF
// CHECK:       ; Data (chains)
// CHECK:       shared:
// CHECK-NEXT:    UC_DMA_BD       0, 0x000010e1, @shared_data, 1, 0, 0
// CHECK:       ; Data
// CHECK:       shared_data:
// CHECK-NEXT:    .long           0x00000001

// -----

// Default-group jobs and attach_to_group(0) jobs use the same chain copy.
// Definitions may follow their users, and group 2 needs its own payload copy.
module {
  aie.device(npu3) {
    aiex.cert.job(3) {
      aiex.cert.uc_dma_write_des_sync(@default_shared)
    }
    aiex.cert.attach_to_group(0) {
      aiex.cert.job(4) {
        aiex.cert.uc_dma_write_des_sync(@default_shared)
      }
    }
    aiex.cert.attach_to_group(2) {
      aiex.cert.job(5) {
        aiex.cert.uc_dma_write_des_sync(@default_shared)
      }
    }
    memref.global "private" constant @default_data : memref<1xi32> = dense<3>
    aiex.cert.uc_dma_chain @default_shared {
      aiex.cert.uc_dma_bd @default_data, 4323, 1, false
    }
  }
}

// CHECK-LABEL: .attach_to_group 0
// CHECK:       START_JOB 3
// CHECK:       START_JOB 4
// CHECK:       default_shared:
// CHECK-NEXT:    UC_DMA_BD       0, 0x000010e3, @default_data, 1, 0, 0
// CHECK-NOT:   default_shared:
// CHECK:       default_data:
// CHECK-NEXT:    .long           0x00000003
// CHECK-NOT:   default_data:
// CHECK-LABEL: .attach_to_group 2
// CHECK:       START_JOB 5
// CHECK:       default_shared:
// CHECK-NEXT:    UC_DMA_BD       0, 0x000010e3, @default_data, 1, 0, 0
// CHECK:       default_data:
// CHECK-NEXT:    .long           0x00000003

// -----

// A group reference must remain visible to chain merging: the adjacent
// device-level writes cannot absorb and erase a chain used by another uC.
module {
  aie.device(npu3) {
    memref.global "private" constant @merge_data1 : memref<1xi32> = dense<4>
    memref.global "private" constant @merge_data2 : memref<1xi32> = dense<5>
    aiex.cert.uc_dma_chain @merge_shared {
      aiex.cert.uc_dma_bd @merge_data1, 4324, 1, false
    }
    aiex.cert.uc_dma_chain @merge_other {
      aiex.cert.uc_dma_bd @merge_data2, 4325, 1, false
    }
    aiex.cert.page {
      aiex.cert.job(6) {
        aiex.cert.uc_dma_write_des_sync(@merge_shared)
        aiex.cert.uc_dma_write_des_sync(@merge_other)
      }
    }
    aiex.cert.attach_to_group(1) {
      aiex.cert.job(7) {
        aiex.cert.uc_dma_write_des_sync(@merge_shared)
      }
    }
  }
}

// CHECK-LABEL: .attach_to_group 0
// CHECK:       merge_shared:
// CHECK:       merge_other:
// CHECK-LABEL: .attach_to_group 1
// CHECK:       START_JOB 7
// CHECK:       merge_shared:
// CHECK-NEXT:    UC_DMA_BD       0, 0x000010e4, @merge_data1, 1, 0, 0
// CHECK:       merge_data1:
// CHECK-NEXT:    .long           0x00000004

// OPT-LABEL: START_JOB 6
// OPT:         uC_DMA_WRITE_DES_SYNC  @merge_shared
// OPT-NEXT:    uC_DMA_WRITE_DES_SYNC  @merge_other
// OPT:       merge_shared:
// OPT:       merge_other:
// OPT-LABEL: .attach_to_group 1
// OPT:       START_JOB 7
// OPT:         uC_DMA_WRITE_DES_SYNC  @merge_shared
// OPT:       merge_shared:
// OPT-NEXT:    UC_DMA_BD       0, 0x000010e4, @merge_data1, 1, 0, 0
// OPT:       merge_data1:
// OPT-NEXT:    .long           0x00000004

//===- cert_isa_ops.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Emitter coverage for the 9 new cert ops. No cert.section here -- a section
// would emit stray .asm files into the CWD.

// RUN: aie-translate -aie-cert-to-asm %s | FileCheck %s
// RUN: aie-translate -aie-cert-to-asm %s | FileCheck %s --check-prefix=COL --strict-whitespace

// CHECK: START_JOB 1
// CHECK:   UC_DMA_WRITE_DES       $r2, @chain0
// CHECK:   WAIT_UC_DMA            $r2
// CHECK:   READ_32                $r1, 0x02100000
// CHECK:   READ_32_D              $r0, $r1
// CHECK:   MASK_POLL_32           0x02100000, 0x0000000f, 0x00000001
// CHECK:   SAVE_TIMESTAMPS        7
// CHECK:   SAVE_REGISTER          0x02100000, 9
// CHECK: END_JOB

// COL: {{^}}  UC_DMA_WRITE_DES       $r2, @chain0
// COL: {{^}}  READ_32_D              $r0, $r1
// COL: {{^}}  MASK_POLL_32           0x02100000, 0x0000000f, 0x00000001

module {
  aie.device(npu3) {
    memref.global "private" constant @data0 : memref<8xi32> = dense<0>
    aiex.cert.uc_dma_chain @chain0 {
      aiex.cert.uc_dma_bd @data0, 0x001a05c0, 8, false
    }
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des(2, @chain0)
      aiex.cert.wait_uc_dma(2)
      aiex.cert.read32(1, 0x02100000)
      aiex.cert.read32_d(0, 1)
      aiex.cert.maskpoll32(0x02100000, 0xf, 1)
      aiex.cert.save_timestamps(7)
      aiex.cert.save_register(0x02100000, 9)
    }
  }
}

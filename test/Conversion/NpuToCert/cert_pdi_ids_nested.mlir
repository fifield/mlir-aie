//===- cert_pdi_ids_nested.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aie-cert-assign-pdi-ids: a section's id covers the sections it reaches
// through a nested cert.load_pdi or a cert.preempt save/restore pair. Changing
// one write32 in a nested section changes the id of the section around it, and
// only that one. The nested load_pdi's own id is rewritten to the inner
// section's id. The last case shows that labels are not hashed.

// RUN: aie-opt --split-input-file --pass-pipeline="builtin.module(aie.device(aie-cert-assign-pdi-ids))" %s | FileCheck %s

// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.section @outer
// CHECK: aiex.cert.load_pdi([[INNER:[0-9]+]], @inner)
// CHECK: aiex.cert.load_pdi([[OUTER:[0-9]+]], @outer)
// CHECK: aiex.cert.load_pdi([[PRE:[0-9]+]], @pre)

aie.device(npu3) {
  memref.global "private" constant @d0 : memref<2xi32> = dense<[1, 2]>
  aiex.cert.uc_dma_chain @c0 {
    aiex.cert.uc_dma_bd @d0, 8192, 2, false
  }
  aiex.cert.section @inner {
    aiex.cert.page {
      aiex.cert.job(7) {
        aiex.cert.write32(4096, 1)
        aiex.cert.uc_dma_write_des_sync(@c0)
      }
    }
  }
  aiex.cert.section @outer {
    aiex.cert.page {
      aiex.cert.job(9) {
        aiex.cert.load_pdi(5, @inner)
      }
    }
  }
  aiex.cert.section @sv {
    aiex.cert.page {
      aiex.cert.job(3) {
        aiex.cert.write32(1, 1)
      }
    }
  }
  aiex.cert.section @rs {
    aiex.cert.page {
      aiex.cert.job(4) {
        aiex.cert.write32(1, 1)
      }
    }
  }
  aiex.cert.section @pre {
    aiex.cert.page {
      aiex.cert.job(5) {
        aiex.cert.preempt(1, @sv, @rs)
      }
    }
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.load_pdi(1, @outer)
    }
  }
  aiex.cert.page {
    aiex.cert.job(2) {
      aiex.cert.load_pdi(1, @pre)
    }
  }
}

// -----

// @inner's write32 value differs: @inner's and @outer's ids change, @pre's
// does not. (CHECK-NOT between CHECK and CHECK-SAME rejects the old id on that
// very line.)
// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.load_pdi
// CHECK-NOT: ([[INNER]],
// CHECK-SAME: @inner)
// CHECK: aiex.cert.load_pdi
// CHECK-NOT: ([[OUTER]],
// CHECK-SAME: @outer)
// CHECK: aiex.cert.load_pdi([[PRE]], @pre)

aie.device(npu3) {
  memref.global "private" constant @d0 : memref<2xi32> = dense<[1, 2]>
  aiex.cert.uc_dma_chain @c0 {
    aiex.cert.uc_dma_bd @d0, 8192, 2, false
  }
  aiex.cert.section @inner {
    aiex.cert.page {
      aiex.cert.job(7) {
        aiex.cert.write32(4096, 2)
        aiex.cert.uc_dma_write_des_sync(@c0)
      }
    }
  }
  aiex.cert.section @outer {
    aiex.cert.page {
      aiex.cert.job(9) {
        aiex.cert.load_pdi(5, @inner)
      }
    }
  }
  aiex.cert.section @sv {
    aiex.cert.page {
      aiex.cert.job(3) {
        aiex.cert.write32(1, 1)
      }
    }
  }
  aiex.cert.section @rs {
    aiex.cert.page {
      aiex.cert.job(4) {
        aiex.cert.write32(1, 1)
      }
    }
  }
  aiex.cert.section @pre {
    aiex.cert.page {
      aiex.cert.job(5) {
        aiex.cert.preempt(1, @sv, @rs)
      }
    }
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.load_pdi(1, @outer)
    }
  }
  aiex.cert.page {
    aiex.cert.job(2) {
      aiex.cert.load_pdi(1, @pre)
    }
  }
}

// -----

// The restore section's write32 value differs: @pre's id changes, @outer's
// does not.
// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.load_pdi([[INNER]], @inner)
// CHECK: aiex.cert.load_pdi([[OUTER]], @outer)
// CHECK: aiex.cert.load_pdi
// CHECK-NOT: ([[PRE]],
// CHECK-SAME: @pre)

aie.device(npu3) {
  memref.global "private" constant @d0 : memref<2xi32> = dense<[1, 2]>
  aiex.cert.uc_dma_chain @c0 {
    aiex.cert.uc_dma_bd @d0, 8192, 2, false
  }
  aiex.cert.section @inner {
    aiex.cert.page {
      aiex.cert.job(7) {
        aiex.cert.write32(4096, 1)
        aiex.cert.uc_dma_write_des_sync(@c0)
      }
    }
  }
  aiex.cert.section @outer {
    aiex.cert.page {
      aiex.cert.job(9) {
        aiex.cert.load_pdi(5, @inner)
      }
    }
  }
  aiex.cert.section @sv {
    aiex.cert.page {
      aiex.cert.job(3) {
        aiex.cert.write32(1, 1)
      }
    }
  }
  aiex.cert.section @rs {
    aiex.cert.page {
      aiex.cert.job(4) {
        aiex.cert.write32(1, 2)
      }
    }
  }
  aiex.cert.section @pre {
    aiex.cert.page {
      aiex.cert.job(5) {
        aiex.cert.preempt(1, @sv, @rs)
      }
    }
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.load_pdi(1, @outer)
    }
  }
  aiex.cert.page {
    aiex.cert.job(2) {
      aiex.cert.load_pdi(1, @pre)
    }
  }
}

// -----

// Labels are not hashed: sections with equal contents share an id whatever
// they are called.
// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.load_pdi([[ID:[0-9]+]], @hw_a)
// CHECK: aiex.cert.load_pdi([[ID]], @hw_b)
// CHECK: aiex.cert.load_pdi([[ID]], @gen_x)
// CHECK: aiex.cert.load_pdi([[ID]], @gen_y)
// CHECK: aiex.cert.load_pdi([[ID]], @dev)

aie.device(npu3) {
  aiex.cert.section @hw_a {
    aiex.cert.page { aiex.cert.job(3) { aiex.cert.write32(4096, 1) } }
  }
  aiex.cert.section @hw_b {
    aiex.cert.page { aiex.cert.job(3) { aiex.cert.write32(4096, 1) } }
  }
  aiex.cert.section @gen_x {
    aiex.cert.page { aiex.cert.job(3) { aiex.cert.write32(4096, 1) } }
  }
  aiex.cert.section @gen_y {
    aiex.cert.page { aiex.cert.job(3) { aiex.cert.write32(4096, 1) } }
  }
  aiex.cert.section @dev {
    aiex.cert.page { aiex.cert.job(3) { aiex.cert.write32(4096, 1) } }
  }
  aiex.cert.page { aiex.cert.job(1) { aiex.cert.load_pdi(1, @hw_a) } }
  aiex.cert.page { aiex.cert.job(2) { aiex.cert.load_pdi(1, @hw_b) } }
  aiex.cert.page { aiex.cert.job(4) { aiex.cert.load_pdi(1, @gen_x) } }
  aiex.cert.page { aiex.cert.job(5) { aiex.cert.load_pdi(1, @gen_y) } }
  aiex.cert.page { aiex.cert.job(6) { aiex.cert.load_pdi(1, @dev) } }
}

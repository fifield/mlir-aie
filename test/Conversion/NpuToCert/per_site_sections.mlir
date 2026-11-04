//===- per_site_sections.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -cert-legalize-pages -aie-cert-per-site-sections --split-input-file --verify-diagnostics %s | FileCheck %s

// Every cert.load_pdi site gets its own copy of the section it loads,
// labelled <lower-case name>__site<n>, in program order. The copies are
// cloned from the template after its job ids are renumbered 1..n. The
// template is erased.

// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.load_pdi(1, @cfg__site0)
// CHECK: aiex.cert.load_pdi(2, @other__site0)
// CHECK: aiex.cert.load_pdi(1, @cfg__site1)
// CHECK-NOT: aiex.cert.section @cfg {

// Both copies are identical: jobs 1 and 2, and the template's payload.
// CHECK-LABEL: aiex.cert.section @cfg__site0 {
// CHECK-NEXT: aiex.cert.page {
// CHECK-NEXT: aiex.cert.job(1) {
// CHECK-NEXT: aiex.cert.write32(4096, 1)
// CHECK: aiex.cert.job(2) {
// CHECK-NEXT: aiex.cert.write32(4096, 2)
// CHECK-LABEL: aiex.cert.section @cfg__site1 {
// CHECK-NEXT: aiex.cert.page {
// CHECK-NEXT: aiex.cert.job(1) {
// CHECK-NEXT: aiex.cert.write32(4096, 1)
// CHECK: aiex.cert.job(2) {
// CHECK-NEXT: aiex.cert.write32(4096, 2)
// CHECK: aiex.cert.section @other__site0 {
// CHECK-NEXT: aiex.cert.page {
// CHECK-NEXT: aiex.cert.job(1) {
// CHECK-NEXT: aiex.cert.write32(8192, 3)
// CHECK-NOT: aiex.cert.section
module {
  aie.device(npu3) @main {
    aiex.cert.job(3) {
      aiex.cert.load_pdi(1, @cfg)
    }
    aiex.cert.job(4) {
      aiex.cert.load_pdi(2, @other)
    }
    aiex.cert.job(5) {
      aiex.cert.load_pdi(1, @cfg)
    }
    aiex.cert.section @cfg {
      aiex.cert.page {
        aiex.cert.job(7) {
          aiex.cert.write32(4096, 1)
        }
        aiex.cert.job(9) {
          aiex.cert.write32(4096, 2)
        }
      }
    }
    aiex.cert.section @other {
      aiex.cert.page {
        aiex.cert.job(8) {
          aiex.cert.write32(8192, 3)
        }
      }
    }
  }
}

// -----

// Labels are lower case: aiebu folds them.

// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.load_pdi(1, @cfg__site0)
// CHECK: aiex.cert.section @cfg__site0 {
// CHECK-NOT: aiex.cert.section
module {
  aie.device(npu3) @main {
    aiex.cert.job(3) {
      aiex.cert.load_pdi(1, @CFG)
    }
    aiex.cert.section @CFG {
      aiex.cert.page {
        aiex.cert.job(7) {
          aiex.cert.write32(4096, 1)
        }
      }
    }
  }
}

// -----

// A label already in use gets a suffix: @cfg's first copy cannot be
// @cfg__site0 while a section of that name exists.

// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.load_pdi(1, @cfg__site0_0)
// CHECK: aiex.cert.load_pdi(2, @cfg__site0__site0)
// CHECK: aiex.cert.section @cfg__site0__site0 {
// CHECK: aiex.cert.write32(4096, 9)
// CHECK: aiex.cert.section @cfg__site0_0 {
// CHECK: aiex.cert.write32(4096, 1)
module {
  aie.device(npu3) @main {
    aiex.cert.job(3) {
      aiex.cert.load_pdi(1, @cfg)
    }
    aiex.cert.job(4) {
      aiex.cert.load_pdi(2, @cfg__site0)
    }
    aiex.cert.section @cfg__site0 {
      aiex.cert.page {
        aiex.cert.job(7) {
          aiex.cert.write32(4096, 9)
        }
      }
    }
    aiex.cert.section @cfg {
      aiex.cert.page {
        aiex.cert.job(7) {
          aiex.cert.write32(4096, 1)
        }
      }
    }
  }
}

// -----

// A template that a cert.preempt still names (as its save and restore
// section) stays: the load site gets @cfg__site0, the preempt keeps @cfg.

// CHECK-LABEL: aie.device(npu3)
// CHECK: aiex.cert.load_pdi(1, @cfg__site0)
// CHECK: aiex.cert.preempt(0, @cfg, @cfg)
// CHECK: aiex.cert.section @cfg__site0 {
// CHECK: aiex.cert.section @cfg {
module {
  aie.device(npu3) @main {
    aiex.cert.job(2) {
      aiex.cert.load_pdi(1, @cfg)
    }
    aiex.cert.job(3) {
      aiex.cert.preempt(0, @cfg, @cfg)
    }
    aiex.cert.section @cfg {
      aiex.cert.page {
        aiex.cert.job(7) {
          aiex.cert.write32(4096, 1)
        }
      }
    }
  }
}

// -----

// A section that loads another section is not supported.

module {
  aie.device(npu3) @main {
    aiex.cert.job(3) {
      aiex.cert.load_pdi(1, @outer)
    }
    aiex.cert.section @outer {
      aiex.cert.page {
        aiex.cert.job(7) {
          // expected-error@+1 {{a cert.section that loads another section is not supported with per-site sections}}
          aiex.cert.load_pdi(2, @inner)
        }
      }
    }
    aiex.cert.section @inner {
      aiex.cert.page {
        aiex.cert.job(8) {
          aiex.cert.write32(4096, 1)
        }
      }
    }
  }
}

// -----

// A program on more than one uC is refused (it needs matching per-uC loads).

module {
  aie.device(npu3) @main {
    aiex.cert.job(3) {
      // expected-error@+1 {{per-site sections do not support programs on more than one microcontroller}}
      aiex.cert.load_pdi(1, @cfg)
    }
    aiex.cert.attach_to_group(2) {
      aiex.cert.page {
        aiex.cert.job(5) {
          aiex.cert.write32(4096, 1)
        }
      }
    }
    aiex.cert.section @cfg {
      aiex.cert.page {
        aiex.cert.job(7) {
          aiex.cert.write32(4096, 1)
        }
      }
    }
  }
}

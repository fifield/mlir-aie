//===- cert_pdi_ids_errors.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aie-cert-assign-pdi-ids diagnostics. test-id-bits=2 leaves only ids 1 and 2
// (0 and 3 are reserved), so three different sections loaded on one uC must
// collide. Two sections loaded on the same uC may share a PDI id only if they
// have the same content, or the firmware skips one when it follows the other.
// The firmware caches the last id per uC, so sections loaded on different uCs
// never collide.

// RUN: aie-opt --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(aie.device(aie-cert-assign-pdi-ids{test-id-bits=2}))" %s

// The id width must leave room for a non-reserved id.
// RUN: not aie-opt --split-input-file --pass-pipeline="builtin.module(aie.device(aie-cert-assign-pdi-ids{test-id-bits=1}))" %s 2>&1 | FileCheck %s --check-prefix=BITS
// BITS: test-id-bits must be in [2, 32], got 1

aie.device(npu3) {
  // expected-note @+1 {{@s2 defined here}}
  aiex.cert.section @s2 {
    aiex.cert.page {
      aiex.cert.job(2) {
        aiex.cert.write32(4096, 2)
      }
    }
  }
  // expected-error @+1 {{cert sections @s2 and @s1 are both loaded on uC 0 and hash to the same PDI id 0x00000002 but differ in content}}
  aiex.cert.section @s1 {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.write32(4096, 1)
      }
    }
  }
  // expected-error @+1 {{cert sections @s2 and @s3 are both loaded on uC 0 and hash to the same PDI id 0x00000002 but differ in content}}
  aiex.cert.section @s3 {
    aiex.cert.page {
      aiex.cert.job(3) {
        aiex.cert.write32(4096, 4)
      }
    }
  }
  aiex.cert.page {
    aiex.cert.job(4) {
      aiex.cert.load_pdi(1, @s1)
    }
  }
  aiex.cert.page {
    aiex.cert.job(5) {
      aiex.cert.load_pdi(2, @s2)
    }
  }
  aiex.cert.page {
    aiex.cert.job(6) {
      aiex.cert.load_pdi(3, @s3)
    }
  }
}

// -----

// Only sections some cert.load_pdi names need distinct ids. Save/restore
// sections reachable only through cert.preempt carry no id the firmware
// compares, so the same four sections that would collide as loaded PDIs are
// fine here: just @pre gets an id.
aie.device(npu3) {
  aiex.cert.section @sv1 {
    aiex.cert.page { aiex.cert.job(1) { aiex.cert.write32(4096, 1) } }
  }
  aiex.cert.section @rs1 {
    aiex.cert.page { aiex.cert.job(2) { aiex.cert.write32(4096, 2) } }
  }
  aiex.cert.section @sv2 {
    aiex.cert.page { aiex.cert.job(3) { aiex.cert.write32(4096, 3) } }
  }
  aiex.cert.section @rs2 {
    aiex.cert.page { aiex.cert.job(4) { aiex.cert.write32(4096, 4) } }
  }
  aiex.cert.section @pre {
    aiex.cert.page { aiex.cert.job(5) { aiex.cert.preempt(1, @sv1, @rs1) } }
    aiex.cert.page { aiex.cert.job(6) { aiex.cert.preempt(2, @sv2, @rs2) } }
  }
  aiex.cert.page {
    aiex.cert.job(7) {
      aiex.cert.load_pdi(1, @pre)
    }
  }
}

// -----

// A section's id covers the sections it loads, so a section that loads itself
// has none.
aie.device(npu3) {
  // expected-error @+1 {{cert.section @self references itself}}
  aiex.cert.section @self {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.load_pdi(1, @self)
      }
    }
  }
}

// -----

aie.device(npu3) {
  aiex.cert.page {
    aiex.cert.job(1) {
      // expected-error @+1 {{cert.load_pdi references @missing, which is not a cert.section in this device}}
      aiex.cert.load_pdi(1, @missing)
    }
  }
}

// -----

// Three sections that all fold to id 1, each loaded on its own uC: no two of
// them share an id on one uC.
aie.device(npu3) {
  aiex.cert.section @s1 {
    aiex.cert.page { aiex.cert.job(1) { aiex.cert.write32(4096, 1) } }
  }
  aiex.cert.section @s2 {
    aiex.cert.page { aiex.cert.job(2) { aiex.cert.write32(4096, 2) } }
  }
  aiex.cert.section @s3 {
    aiex.cert.page { aiex.cert.job(3) { aiex.cert.write32(4096, 3) } }
  }
  aiex.cert.page { aiex.cert.job(4) { aiex.cert.load_pdi(1, @s1) } }
  aiex.cert.attach_to_group(1) {
    aiex.cert.page { aiex.cert.job(5) { aiex.cert.load_pdi(1, @s2) } }
  }
  aiex.cert.attach_to_group(2) {
    aiex.cert.page { aiex.cert.job(6) { aiex.cert.load_pdi(1, @s3) } }
  }
}

// -----

// A section loaded from inside another section is loaded on that section's
// uCs: @outer is loaded on uC 1 and loads @s1 and @s2, so all three are on
// uC 1 and two of them collide there.
aie.device(npu3) {
  // expected-note @+1 {{@s1 defined here}}
  aiex.cert.section @s1 {
    aiex.cert.page { aiex.cert.job(1) { aiex.cert.write32(4096, 1) } }
  }
  // expected-error @+1 {{cert sections @s1 and @s2 are both loaded on uC 1 and hash to the same PDI id 0x00000002}}
  aiex.cert.section @s2 {
    aiex.cert.page { aiex.cert.job(2) { aiex.cert.write32(4096, 2) } }
  }
  aiex.cert.section @outer {
    aiex.cert.page { aiex.cert.job(3) { aiex.cert.load_pdi(1, @s1) } }
    aiex.cert.page { aiex.cert.job(4) { aiex.cert.load_pdi(1, @s2) } }
  }
  aiex.cert.attach_to_group(1) {
    aiex.cert.page { aiex.cert.job(5) { aiex.cert.load_pdi(1, @outer) } }
  }
}

// -----

// Sections with the same content share an id on one uC without error,
// whatever they are labelled: the firmware may skip reloading one after the
// other, and that is harmless.
aie.device(npu3) {
  aiex.cert.section @site0 {
    aiex.cert.page { aiex.cert.job(1) { aiex.cert.write32(4096, 1) } }
  }
  aiex.cert.section @site1 {
    aiex.cert.page { aiex.cert.job(1) { aiex.cert.write32(4096, 1) } }
  }
  aiex.cert.page { aiex.cert.job(2) { aiex.cert.load_pdi(1, @site0) } }
  aiex.cert.page { aiex.cert.job(3) { aiex.cert.load_pdi(1, @site1) } }
}

// -----

// Placement not yet lowered to cert.attach_to_group: a page's `placement`
// names its uC. The three sections fold to id 1, but each is loaded on its own
// uC.
aie.device(npu3) {
  aiex.cert.section @s1 {
    aiex.cert.page { aiex.cert.job(1) { aiex.cert.write32(4096, 1) } }
  }
  aiex.cert.section @s2 {
    aiex.cert.page { aiex.cert.job(2) { aiex.cert.write32(4096, 2) } }
  }
  aiex.cert.section @s3 {
    aiex.cert.page { aiex.cert.job(3) { aiex.cert.write32(4096, 3) } }
  }
  aiex.cert.page { aiex.cert.job(4) { aiex.cert.load_pdi(1, @s1) } }
  aiex.cert.page { aiex.cert.job(5) { aiex.cert.load_pdi(1, @s2) } } {placement = 1 : i32}
  aiex.cert.page { aiex.cert.job(6) { aiex.cert.load_pdi(1, @s3) } } {placement = 2 : i32}
}

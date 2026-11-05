//===- cert_verify_preempt.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// G-preempt: every uC must expose the same multiset of cert.preempt ids.

// RUN: aie-opt --aie-cert-verify --split-input-file --verify-diagnostics %s

// Every cert.preempt also needs an earlier cert.load_pdi on its uC (rule 3,
// see cert_verify_preempt_rules.mlir), so each uC loads a section first.

// Positive: both uCs expose preempt id 0 -> OK.
aie.device(xcve3858) {
  aiex.cert.attach_to_group(0) {
    aiex.cert.job(0) {
      aiex.cert.load_pdi(7, @cfg)
    }
    aiex.cert.job(1) {
      aiex.cert.preempt(0, @save0, @restore0)
    }
  }
  aiex.cert.attach_to_group(2) {
    aiex.cert.job(2) {
      aiex.cert.load_pdi(8, @cfg_uc2)
    }
    aiex.cert.job(3) {
      aiex.cert.preempt(0, @save0, @restore0)
    }
  }
  aiex.cert.section @cfg {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.write32(0x2000, 42)
      }
    }
  }
  aiex.cert.section @cfg_uc2 {
    aiex.cert.page {
      aiex.cert.job(1) {
      }
    }
  }
}

// -----

// Positive: device-level (uC 0) and group 2 both expose preempt id 0 -> OK.
aie.device(xcve3858) {
  aiex.cert.job(0) {
    aiex.cert.load_pdi(7, @cfg)
  }
  aiex.cert.job(1) {
    aiex.cert.preempt(0, @save0, @restore0)
  }
  aiex.cert.attach_to_group(2) {
    aiex.cert.job(2) {
      aiex.cert.load_pdi(8, @cfg_uc2)
    }
    aiex.cert.job(3) {
      aiex.cert.preempt(0, @save0, @restore0)
    }
  }
  aiex.cert.section @cfg {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.write32(0x2000, 42)
      }
    }
  }
  aiex.cert.section @cfg_uc2 {
    aiex.cert.page {
      aiex.cert.job(1) {
      }
    }
  }
}

// -----

// Negative: uC 0 exposes preempt 0 but group 2 exposes none -> error on group 2.
aie.device(xcve3858) {
  aiex.cert.job(0) {
    aiex.cert.load_pdi(7, @cfg)
  }
  aiex.cert.job(1) {
    aiex.cert.preempt(0, @save0, @restore0)
  }
  // expected-error@+1 {{cert.preempt ids on uC 2 differ from uC 0}}
  aiex.cert.attach_to_group(2) {
    aiex.cert.job(2) {
      aiex.cert.write32(0x2000, 43)
    }
  }
  aiex.cert.section @cfg {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.write32(0x2000, 42)
      }
    }
  }
}

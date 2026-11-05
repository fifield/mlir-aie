//===- cert_verify_preempt_rules.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Preemption rules. Restore brings memtile data back and then replays the
// last section the uC loaded (LOAD_LAST_PDI).

// RUN: aie-opt -aie-cert-verify --verify-diagnostics --split-input-file %s

// Rule 3: a cert.preempt with no earlier load on its uC.
aie.device(npu3) {
  aiex.cert.job(1) {
    // expected-error@+1 {{cert.preempt needs an earlier cert.load_pdi on uC 0: on restore LOAD_LAST_PDI replays the page index of the last load, which is unset or stale without one}}
    aiex.cert.preempt(0, @save0, @restore0)
  }
  aiex.cert.job(2) {
    aiex.cert.load_pdi(7, @cfg)
  }
  aiex.cert.section @cfg {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.write32(0x110000, 1)
      }
    }
  }
}

// -----

// A preempt after the entry load: clean. The section writes memtile (0, 1)
// lock 0 (0x110000) and the last word of its register space (0x1FFFFC,
// offset 0xFFFFC), not data memory.
aie.device(npu3) {
  aiex.cert.job(1) {
    aiex.cert.load_pdi(7, @main__site0)
  }
  aiex.cert.job(2) {
    aiex.cert.preempt(0, @save0, @restore0)
  }
  aiex.cert.section @main__site0 {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.write32(0x110000, 1)
        aiex.cert.write32(0x1FFFFC, 1)
      }
    }
  }
}

// -----

// Rule 1: a loaded section that writes memtile data memory. On AIE4 memtile
// (0, 1)'s data starts 1 MiB into the tile: 0x100000 + 0x100000.
aie.device(npu3) {
  aiex.cert.job(1) {
    aiex.cert.load_pdi(7, @cfg)
  }
  aiex.cert.job(2) {
    aiex.cert.preempt(0, @save0, @restore0)
  }
  aiex.cert.section @cfg {
    aiex.cert.page {
      aiex.cert.job(1) {
        // expected-error@+1 {{section @cfg writes memtile data memory at 0x200000; a design with cert.preempt restores memtile data before replaying this section, which would overwrite it}}
        aiex.cert.write32(0x200000, 1)
      }
    }
  }
}

// -----

// The same through a uC-DMA chain (a buffer initializer): its BD writes 16
// words ending at the last word of the memtile's data memory (0x4FFFFC).
aie.device(npu3) {
  memref.global "private" constant @data : memref<16xi32> = dense<1>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @data, 0x4FFFC0, 16, false
  }
  aiex.cert.job(1) {
    aiex.cert.load_pdi(7, @cfg)
  }
  aiex.cert.job(2) {
    aiex.cert.preempt(0, @save0, @restore0)
  }
  aiex.cert.section @cfg {
    aiex.cert.page {
      aiex.cert.job(1) {
        // expected-error@+1 {{section @cfg writes memtile data memory at 0x4FFFC0}}
        aiex.cert.uc_dma_write_des_sync(@chain)
      }
    }
  }
}

// -----

// A cert.write32_d with an immediate address is checked like cert.write32.
// One with address_is_reg takes its address from a register at run time
// (here read from memtile (0, 1)'s register space, but it could be anything),
// so it cannot be shown to miss memtile data memory and is rejected.
aie.device(npu3) {
  aiex.cert.job(1) {
    aiex.cert.load_pdi(7, @cfg)
  }
  aiex.cert.job(2) {
    aiex.cert.preempt(0, @save0, @restore0)
  }
  aiex.cert.section @cfg {
    aiex.cert.page {
      aiex.cert.job(1) {
        // expected-error@+1 {{section @cfg writes memtile data memory at 0x200004}}
        aiex.cert.write32_d(0x200004, 1)
        aiex.cert.write32_d(0x110000, 1)
        aiex.cert.read32(0, 0x110004)
        // expected-error@+1 {{section @cfg has a cert.write32_d whose address is in register $r0, which may be in memtile data memory; a design with cert.preempt restores memtile data before replaying this section, which would overwrite it}}
        aiex.cert.write32_d(0, 1) {address_is_reg}
      }
    }
  }
}

// -----

// Without a cert.preempt, the same section is clean; and a section only a
// cert.preempt names (save/restore code) is not a loaded configuration.
aie.device(npu3) {
  aiex.cert.job(1) {
    aiex.cert.load_pdi(7, @cfg)
  }
  aiex.cert.section @cfg {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.write32(0x200000, 1)
      }
    }
  }
}

// -----

aie.device(npu3) {
  aiex.cert.job(1) {
    aiex.cert.load_pdi(7, @cfg)
  }
  aiex.cert.job(2) {
    aiex.cert.preempt(0, @save0, @restore0)
  }
  aiex.cert.section @cfg {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.write32(0x110000, 1)
      }
    }
  }
  aiex.cert.section @restore0 {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.write32(0x200000, 1)
      }
    }
  }
}

// -----

// Rule 3 is per uC: a load on uC 0 does not cover a preempt on uC 2.
aie.device(npu3) {
  aiex.cert.job(1) {
    aiex.cert.load_pdi(7, @cfg)
  }
  aiex.cert.job(2) {
    aiex.cert.preempt(0, @save0, @restore0)
  }
  aiex.cert.attach_to_group(2) {
    aiex.cert.job(1) {
      // expected-error@+1 {{cert.preempt needs an earlier cert.load_pdi on uC 2}}
      aiex.cert.preempt(0, @save0, @restore0)
    }
  }
  aiex.cert.section @cfg {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.write32(0x110000, 1)
      }
    }
  }
}

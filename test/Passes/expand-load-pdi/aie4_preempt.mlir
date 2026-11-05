//===- aie4_preempt.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// On restore, LOAD_LAST_PDI replays only the last loaded section, not the
// configuration written in line after it, so a device with a cert.preempt
// cannot have its loads expanded.

// RUN: aie-opt --aie-expand-load-pdi --verify-diagnostics %s

module {
  aie.device(npu3_1col) @a {
    %t = aie.tile(0, 2)
    %l = aie.lock(%t, 0) {init = 1 : i32}
  }
  aie.device(npu3_1col) @main {
    %t = aie.tile(0, 0)
    aiex.cert.job(3) {
      // expected-error@+1 {{cert.preempt cannot be combined with configuration written in line (--expand-load-pdis / --load-pdi-to-ctrl-pkt): restore replays only the last loaded section}}
      aiex.cert.preempt(0, @save0, @restore0)
    }
    aie.runtime_sequence @seq(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @a}
    }
  }
}

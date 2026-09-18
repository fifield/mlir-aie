// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// Crossbars cannot connect ports on different switches, even at a matching
// bridge index. Crossing must use an output -> input bridge wire.
module {
  aie.device(npu3_1col) {
    %mem = aie.tile(0, 1)
    aie.switchbox(%mem) {
      // expected-error @+1 {{illegal stream switch connection}}
      aie.connect<Switch32 : 0, North32 : 0>
    }
  }
}

// -----

module {
  aie.device(npu3_1col) {
    %core = aie.tile(0, 2)
    aie.switchbox(%core) {
      // expected-error @+1 {{illegal stream switch connection}}
      aie.connect<Switch32 : 0, Switch512 : 0>
    }
  }
}

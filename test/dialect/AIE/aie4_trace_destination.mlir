// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --split-input-file --verify-diagnostics

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.switchbox(%shim) {
      aie.connect<Switch32 : 0, DMATrace : 0>
    }
  }
}

// -----

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.switchbox(%shim) {
      // expected-error@+1 {{index 1 for dest bundle DMATrace must be less than 1}}
      aie.connect<North : 0, DMATrace : 1>
    }
  }
}

// -----

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.switchbox(%shim) {
      // expected-error@+1 {{source bundle DMATrace not supported}}
      aie.connect<DMATrace : 0, North : 0>
    }
  }
}

// -----

module {
  aie.device(npu3) {
    %mem = aie.tile(0, 1)
    aie.switchbox(%mem) {
      // expected-error@+1 {{dest bundle DMATrace not supported}}
      aie.connect<North : 0, DMATrace : 0>
    }
  }
}

// -----

module {
  aie.device(npu1) {
    %shim = aie.tile(0, 0)
    aie.switchbox(%shim) {
      // expected-error@+1 {{dest bundle DMATrace not supported}}
      aie.connect<North : 0, DMATrace : 0>
    }
  }
}

// -----

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.switchbox(%shim) {
      // expected-error@+1 {{illegal stream switch connection}}
      aie.connect<DMA : 0, DMATrace : 0>
    }
  }
}

// -----

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.switchbox(%shim) {
      // expected-error@+1 {{illegal stream switch connection}}
      aie.connect<North32 : 0, DMATrace : 0>
    }
  }
}

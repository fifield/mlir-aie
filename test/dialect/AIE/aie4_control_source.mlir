// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --split-input-file --verify-diagnostics

aie.device(npu3) {
  %tile = aie.tile(0, 0)
  aie.switchbox(%tile) {
    aie.connect<DMAControl : 0, North : 0>
    aie.end
  }
}

// -----

aie.device(npu3) {
  %tile = aie.tile(0, 0)
  aie.switchbox(%tile) {
    aie.connect<DMAControl : 1, North : 2>
    aie.end
  }
}

// -----

aie.device(npu3) {
  %tile = aie.tile(0, 0)
  aie.switchbox(%tile) {
    // expected-error@+1 {{index 2 for source bundle DMAControl must be less than 2}}
    aie.connect<DMAControl : 2, North : 0>
    aie.end
  }
}

// -----

aie.device(npu3) {
  %tile = aie.tile(0, 0)
  aie.switchbox(%tile) {
    // expected-error@+1 {{dest bundle DMAControl not supported}}
    aie.connect<North : 0, DMAControl : 0>
    aie.end
  }
}

// -----

aie.device(npu3) {
  %tile = aie.tile(0, 2)
  aie.switchbox(%tile) {
    // expected-error@+1 {{source bundle DMAControl not supported}}
    aie.connect<DMAControl : 0, North : 0>
    aie.end
  }
}

// -----

aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.switchbox(%tile) {
    // expected-error@+1 {{source bundle DMAControl not supported}}
    aie.connect<DMAControl : 0, North : 0>
    aie.end
  }
}

// -----

aie.device(npu3) {
  %tile = aie.tile(0, 0)
  aie.switchbox(%tile) {
    // expected-error@+1 {{source bundle TileControl not supported}}
    aie.connect<TileControl : 0, North : 0>
    aie.end
  }
}

// -----

aie.device(npu3) {
  %tile = aie.tile(0, 0)
  aie.switchbox(%tile) {
    // expected-error@+1 {{illegal stream switch connection}}
    aie.connect<DMAControl : 0, DMATrace : 0>
    aie.end
  }
}

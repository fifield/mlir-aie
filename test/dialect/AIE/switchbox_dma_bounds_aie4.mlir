// RUN: aie-opt --split-input-file --verify-diagnostics %s -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Check DMA source and destination bounds independently.
module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    aie.switchbox(%shim) {
      aie.connect<DMA : 3, North : 0>
      aie.connect<North : 1, DMA : 1>
    }
    aie.switchbox(%mem) {
      aie.connect<DMA : 9, North : 4>
      aie.connect<North : 3, DMA : 7>
    }
  }
}

// -----

// Memtile MM2S channels 8 and 9 do not imply corresponding S2MM channels.
module {
  aie.device(npu3) {
    %mem = aie.tile(0, 1)
    aie.switchbox(%mem) {
      // expected-error@+1 {{index 8 for dest bundle DMA must be less than 8}}
      aie.connect<Switch32 : 0, DMA : 8>
    }
  }
}

// -----

// Four shim MM2S streams do not imply four application S2MM channels.
module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.switchbox(%shim) {
      // expected-error@+1 {{index 2 for dest bundle DMA must be less than 2}}
      aie.connect<North : 0, DMA : 2>
    }
  }
}

//===- npu3_shim_mux.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// AIE4 has no shim mux: shim DMA, NOC and PLIO endpoints are switchbox ports.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    // expected-error@+1 {{'aie.shim_mux' op tile (0, 0) has no shim mux: its DMA connects to the switchbox directly}}
    aie.shim_mux(%shim) {
      aie.connect<DMA : 0, North : 3>
    }
  }
}

// -----

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    // expected-error@+1 {{'aie.flow' op source NOC:0 does not exist: tile (0, 0) has 0 NOC ports into its stream switch}}
    aie.flow(%shim, NOC : 0, %core, DMA : 0)
  }
}

// -----

module {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    // expected-error@+1 {{'aie.flow' op destination PLIO:0 does not exist: tile (0, 0) has 0 PLIO ports out of its stream switch}}
    aie.flow(%core, DMA : 0, %shim, PLIO : 0)
  }
}

// -----

// A target with a shim mux keeps its NOC and PLIO ports there.
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.flow(%shim, NOC : 0, %core, DMA : 0)
    aie.flow(%core, DMA : 0, %shim, PLIO : 0)
  }
}

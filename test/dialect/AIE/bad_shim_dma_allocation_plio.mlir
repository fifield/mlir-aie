//===- bad_shim_dma_allocation_plio.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A PLIO allocation index is a PL stream port, bounded by the shim mux PLIO
// ports (the switchbox's on AIE4, which has no shim mux).

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// An AIE2 shim mux has 8 PLIO ports into the array and 6 out of it.
aie.device(xcve2302) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @in(%shim, MM2S, 7) {plio = true}
  aie.shim_dma_allocation @out(%shim, S2MM, 5) {plio = true}
}

// -----

aie.device(xcve2302) {
  %shim = aie.tile(0, 0)
  // expected-error@+1 {{PLIO channel 8 is out of range: the tile has 8 MM2S PLIO ports}}
  aie.shim_dma_allocation @in(%shim, MM2S, 8) {plio = true}
}

// -----

aie.device(xcve2302) {
  %shim = aie.tile(0, 0)
  // expected-error@+1 {{PLIO channel 6 is out of range: the tile has 6 S2MM PLIO ports}}
  aie.shim_dma_allocation @out(%shim, S2MM, 6) {plio = true}
}

// -----

// The shim's DMA channel count does not bound the PLIO index.
aie.device(xcve2302) {
  %shim = aie.tile(2, 0)
  aie.shim_dma_allocation @in(%shim, MM2S, 4) {plio = true}
}

// -----

// An AIE4 shim has no shim mux, and its switchbox has no PLIO ports.
aie.device(npu3) {
  %shim = aie.tile(0, 0)
  // expected-error@+1 {{PLIO channel 0 is out of range: the tile has 0 MM2S PLIO ports}}
  aie.shim_dma_allocation @in(%shim, MM2S, 0) {plio = true}
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  // expected-error@+1 {{a PLIO allocation must be MM2S or S2MM}}
  aie.shim_dma_allocation @ctrl(%shim, ControlMM2S, 0) {plio = true}
}

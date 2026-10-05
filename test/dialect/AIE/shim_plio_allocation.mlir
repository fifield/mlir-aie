// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// PLIO stream metadata is valid on shim-PL and shim-NOC tiles. Its port
// bounds differ from memory DMA: eight MM2S and six S2MM PLIO ports on AIE2.
aie.device(xcve2302) {
  %pl = aie.tile(1, 0)
  %noc = aie.tile(2, 0)
  aie.shim_dma_allocation @pl_in (%pl, MM2S, 7) {plio = true}
  aie.shim_dma_allocation @pl_out (%pl, S2MM, 5) {plio = true}
  aie.shim_dma_allocation @noc_in (%noc, MM2S, 7) {plio = true}
  aie.shim_dma_allocation @noc_out (%noc, S2MM, 5) {plio = true}
}

// -----

aie.device(xcve2302) {
  %pl = aie.tile(1, 0)
  // expected-error@+1 {{PLIO channel 8 is out of range; expected 0 <= channel < 8}}
  aie.shim_dma_allocation @bad (%pl, MM2S, 8) {plio = true}
}

// -----

aie.device(xcve2302) {
  %pl = aie.tile(1, 0)
  // expected-error@+1 {{PLIO channel 6 is out of range; expected 0 <= channel < 6}}
  aie.shim_dma_allocation @bad (%pl, S2MM, 6) {plio = true}
}

// -----

aie.device(xcve2302) {
  %pl = aie.tile(1, 0)
  // expected-error@+1 {{PLIO channel -1 is out of range; expected 0 <= channel < 8}}
  aie.shim_dma_allocation @bad (%pl, MM2S, -1) {plio = true}
}

// -----

// PLIO metadata must not admit specialized memory-DMA engines.
aie.device(npu3) {
  %noc = aie.tile(0, 0)
  // expected-error@+1 {{PLIO allocations only support MM2S or S2MM}}
  aie.shim_dma_allocation @bad (%noc, ControlMM2S, 0) {plio = true}
}

// -----

// A memory-DMA allocation still cannot use a shim-PL tile.
aie.device(xcve2302) {
  %pl = aie.tile(1, 0)
  // expected-error@+1 {{unsupported DMA channel (MM2S, 0) on tile (1, 0)}}
  aie.shim_dma_allocation @bad (%pl, MM2S, 0)
}

// -----

// A PLIO allocation does not make memory-DMA lowering available on shim-PL.
aie.device(xcve2302) {
  %pl = aie.tile(1, 0)
  aie.shim_dma_allocation @stream (%pl, MM2S, 0) {plio = true}
  aie.runtime_sequence(%arg: memref<64xi32>) {
    // expected-error@+1 {{unsupported DMA channel (MM2S, 0) on tile (1, 0)}}
    aiex.npu.dma_memcpy_nd (%arg[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @stream, id = 0 : i64} : memref<64xi32>
  }
}

//===- dma_channel_reset_invalid.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -split-input-file -verify-diagnostics --aie-lower-dma-channel-reset %s

// A channel index beyond the tile's DMA channel count is rejected by the
// verifier before it can reach getDmaControlAddress and emit a write to an
// unintended register. A core tile has 2 S2MM channels, so channel 2 is out of
// range.
module {
  aie.device(npu2) {
    %core_tile = aie.tile(0, 3)
    aie.runtime_sequence() {
      // expected-error @+1 {{channel 2 out of range for this tile and direction}}
      aiex.dma_channel_reset(%core_tile, S2MM, 2)
    }
  }
}

// -----

// A shim NOC tile is rejected: only core and mem tiles have a per-channel DMA
// reset bit. On the shim NOC DMA control register bit 1 is PAUSE_MEM, not RESET
// (aie-rt's Aie2PShimDmaChProp sets Reset.Mask = 0), so there is nothing valid
// to lower to. This matches aie-rt's XAie_DmaChannelReset, which errors on
// SHIMNOC/SHIMPL tiles.
module {
  aie.device(npu2) {
    %shim_tile = aie.tile(0, 0)
    aie.runtime_sequence() {
      // expected-error @+1 {{has no DMA channel reset (only core and mem tiles do; shim NOC DMA has no reset bit)}}
      aiex.dma_channel_reset(%shim_tile, S2MM, 0)
    }
  }
}

// -----

// Specialized shim engines do not exist on core or memory tiles.
aie.device(npu2) {
  %tile = aie.tile(0, 2)
  aie.runtime_sequence() {
    // expected-error@+1 {{unsupported DMA channel (ControlMM2S, 0) on tile (0, 2)}}
    aiex.dma_channel_reset(%tile, ControlMM2S, 0)
  }
}

// -----

aie.device(npu2) {
  %tile = aie.tile(0, 1)
  aie.runtime_sequence() {
    // expected-error@+1 {{unsupported DMA channel (TraceS2MM, 0) on tile (0, 1)}}
    aiex.dma_channel_reset(%tile, TraceS2MM, 0)
  }
}

// -----

aie.device(npu3) {
  %tile = aie.tile(0, 1)
  aie.runtime_sequence() {
    // expected-error@+1 {{unsupported DMA channel (ControlMM2S, 0) on tile (0, 1)}}
    aiex.dma_channel_reset(%tile, ControlMM2S, 0)
  }
}

// -----

aie.device(npu3) {
  %tile = aie.tile(0, 2)
  aie.runtime_sequence() {
    // expected-error@+1 {{unsupported DMA channel (TraceS2MM, 0) on tile (0, 2)}}
    aiex.dma_channel_reset(%tile, TraceS2MM, 0)
  }
}

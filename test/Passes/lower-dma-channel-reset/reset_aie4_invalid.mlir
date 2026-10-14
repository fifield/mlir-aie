// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt -split-input-file -verify-diagnostics %s

aie.device(npu3) {
  %t = aie.tile(0, 1)
  aie.runtime_sequence() {
    // expected-error@+1 {{has no core to reset}}
    aiex.core_reset(%t)
  }
}

// -----

aie.device(npu3) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence() {
    // expected-error@+1 {{has no core to reset}}
    aiex.core_reset(%t)
  }
}

// -----

aie.device(npu3) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence() {
    // expected-error@+1 {{has no DMA channel reset}}
    aiex.dma_channel_reset(%t, S2MM, 0)
  }
}

// -----

aie.device(npu3) {
  %t = aie.tile(0, 2)
  aie.runtime_sequence() {
    // expected-error@+1 {{channel 2 out of range}}
    aiex.dma_channel_reset(%t, S2MM, 2)
  }
}

// -----

aie.device(npu3) {
  %t = aie.tile(0, 2)
  aie.runtime_sequence() {
    // expected-error@+1 {{channel 1 out of range}}
    aiex.dma_channel_reset(%t, MM2S, 1)
  }
}

// -----

aie.device(npu3) {
  %t = aie.tile(0, 1)
  aie.runtime_sequence() {
    // expected-error@+1 {{channel 8 out of range}}
    aiex.dma_channel_reset(%t, S2MM, 8)
  }
}

// -----

aie.device(npu3) {
  %t = aie.tile(0, 1)
  aie.runtime_sequence() {
    // expected-error@+1 {{channel 10 out of range}}
    aiex.dma_channel_reset(%t, MM2S, 10)
  }
}

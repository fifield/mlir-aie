// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-lower-core-reset --aie-lower-dma-channel-reset --aie-npu-to-cert %s | FileCheck %s --implicit-check-not=aiex.npu --implicit-check-not=aiex.cert.write32

// Pin the final CERT addresses and reset-only masks. AIE4 compute row 2 is
// physical address row 6 (the memtile occupies five rows); column 2 and core
// row 5 also exercise column addressing. Cover every compute and memtile DMA
// channel, including both memtile banks and the asymmetric channel counts.
module {
  aie.device(npu3) {
    %core = aie.tile(0, 2)
    %other = aie.tile(2, 5)
    %mem = aie.tile(0, 1)
    aie.runtime_sequence() {
      // CHECK: aiex.cert.maskwrite32(6959104, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(6959104, 2, 0)
      aiex.core_reset(%core)

      // CHECK: aiex.cert.maskwrite32(77213696, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(77213696, 2, 0)
      aiex.core_reset(%other)

      // CHECK: aiex.cert.maskwrite32(6999552, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(6999552, 2, 0)
      aiex.dma_channel_reset(%core, S2MM, 0)

      // CHECK: aiex.cert.maskwrite32(6999560, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(6999560, 2, 0)
      aiex.dma_channel_reset(%core, S2MM, 1)

      // CHECK: aiex.cert.maskwrite32(6999568, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(6999568, 2, 0)
      aiex.dma_channel_reset(%core, MM2S, 0)

      // CHECK: aiex.cert.maskwrite32(1089024, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1089024, 2, 0)
      aiex.dma_channel_reset(%mem, S2MM, 0)

      // CHECK: aiex.cert.maskwrite32(1089032, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1089032, 2, 0)
      aiex.dma_channel_reset(%mem, S2MM, 1)

      // CHECK: aiex.cert.maskwrite32(1089040, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1089040, 2, 0)
      aiex.dma_channel_reset(%mem, S2MM, 2)

      // CHECK: aiex.cert.maskwrite32(1089048, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1089048, 2, 0)
      aiex.dma_channel_reset(%mem, S2MM, 3)

      // CHECK: aiex.cert.maskwrite32(1351168, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1351168, 2, 0)
      aiex.dma_channel_reset(%mem, S2MM, 4)

      // CHECK: aiex.cert.maskwrite32(1351176, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1351176, 2, 0)
      aiex.dma_channel_reset(%mem, S2MM, 5)

      // CHECK: aiex.cert.maskwrite32(1351184, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1351184, 2, 0)
      aiex.dma_channel_reset(%mem, S2MM, 6)

      // CHECK: aiex.cert.maskwrite32(1351192, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1351192, 2, 0)
      aiex.dma_channel_reset(%mem, S2MM, 7)

      // CHECK: aiex.cert.maskwrite32(1089120, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1089120, 2, 0)
      aiex.dma_channel_reset(%mem, MM2S, 0)

      // CHECK: aiex.cert.maskwrite32(1089128, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1089128, 2, 0)
      aiex.dma_channel_reset(%mem, MM2S, 1)

      // CHECK: aiex.cert.maskwrite32(1089136, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1089136, 2, 0)
      aiex.dma_channel_reset(%mem, MM2S, 2)

      // CHECK: aiex.cert.maskwrite32(1089144, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1089144, 2, 0)
      aiex.dma_channel_reset(%mem, MM2S, 3)

      // CHECK: aiex.cert.maskwrite32(1089152, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1089152, 2, 0)
      aiex.dma_channel_reset(%mem, MM2S, 4)

      // CHECK: aiex.cert.maskwrite32(1351264, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1351264, 2, 0)
      aiex.dma_channel_reset(%mem, MM2S, 5)

      // CHECK: aiex.cert.maskwrite32(1351272, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1351272, 2, 0)
      aiex.dma_channel_reset(%mem, MM2S, 6)

      // CHECK: aiex.cert.maskwrite32(1351280, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1351280, 2, 0)
      aiex.dma_channel_reset(%mem, MM2S, 7)

      // CHECK: aiex.cert.maskwrite32(1351288, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1351288, 2, 0)
      aiex.dma_channel_reset(%mem, MM2S, 8)

      // CHECK: aiex.cert.maskwrite32(1351296, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1351296, 2, 0)
      aiex.dma_channel_reset(%mem, MM2S, 9)

    }
  }
}

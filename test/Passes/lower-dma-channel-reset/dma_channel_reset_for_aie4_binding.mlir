// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-lower-dma-channel-reset --aie-dma-to-npu --aie-lower-set-lock --aie-npu-to-cert %s | FileCheck %s --implicit-check-not=aiex.npu --implicit-check-not=aie.objectfifo_rearm_binding

// A preassigned binding exercises the last MM2S memtile channel (bank B),
// S2MM channel 1 on a compute tile, upper lock IDs, and nonzero BD/repeat
// values. Preserve them through reset_for instead of assuming channel/BD 0.
module @bank_b {
  aie.device(npu3) {
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    %ml = aie.lock(%mem, 63) {init = 2 : i32}
    %cl = aie.lock(%core, 15) {init = 0 : i32}
    aie.objectfifo_rearm_binding @r channels(%mem, %core : index, index) locks(%ml, %cl : index, index) {channel_dirs = array<i32: 1, 0>, channel_indices = array<i32: 9, 1>, lock_inits = array<i32: 2, 0>, head_bd_ids = array<i32: 7, 15>, repeat_counts = array<i32: 2, 3>}
    aie.runtime_sequence() {
      // CHECK-LABEL: module @bank_b
      // CHECK: aiex.cert.job
      // CHECK-NEXT: aiex.cert.maskwrite32(1351296, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(1351296, 2, 0)
      // CHECK-NEXT: aiex.cert.maskwrite32(6999560, 2, 2)
      // CHECK-NEXT: aiex.cert.maskwrite32(6999560, 2, 0)
      // CHECK-NEXT: aiex.cert.write32(1376752, 2)
      // CHECK-NEXT: aiex.cert.write32(7004400, 0)
      // CHECK-NEXT: aiex.cert.write32(1351300, 131079)
      // CHECK-NEXT: aiex.cert.write32(6999564, 196623)
      aiex.dma_channel_reset_for(@r)
    }
  }
}

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses --aie-dma-tasks-to-npu --aie-dma-to-npu %s | FileCheck %s

// AIE4 memtile channels have 16 private BDs (0x300 bytes per channel).
// MM2S channels 0..4 / 5..9 and S2MM channels 0..3 / 4..7 use separate
// application banks. Check BD writes and task queues at both bank boundaries.
// In particular, MM2S channel 0 BD4 must not alias channel 1 BD0.

// CHECK-LABEL: @mm2s0_bd4
// CHECK: aiex.npu.blockwrite
// CHECK-SAME: address = 1084608 : ui32
// CHECK: %[[QUEUE:.*]] = arith.constant 1089124 : i32
// CHECK: aiex.npu.write32(%[[QUEUE]], %{{.*}}) : i32, i32
module @mm2s0_bd4 {
  aie.device(npu3) {
    %mem = aie.tile(0, 1)
    %buf = aie.buffer(%mem) : memref<64xi32>
    aie.runtime_sequence() {
      %task = aiex.dma_configure_task(%mem, MM2S, 0) {
        aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64) {bd_id = 4 : i32}
        aie.end
      }
      aiex.dma_start_task(%task)
    }
  }
}

// -----

// CHECK-LABEL: @mm2s1_bd0
// CHECK: aiex.npu.blockwrite
// CHECK-SAME: address = 1085184 : ui32
// CHECK: %[[QUEUE:.*]] = arith.constant 1089132 : i32
// CHECK: aiex.npu.write32(%[[QUEUE]], %{{.*}}) : i32, i32
module @mm2s1_bd0 {
  aie.device(npu3) {
    %mem = aie.tile(0, 1)
    %buf = aie.buffer(%mem) : memref<64xi32>
    aie.runtime_sequence() {
      %task = aiex.dma_configure_task(%mem, MM2S, 1) {
        aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%task)
    }
  }
}

// -----

// CHECK-LABEL: @mm2s4_bd15
// CHECK: aiex.npu.blockwrite
// CHECK-SAME: address = 1088208 : ui32
// CHECK: %[[QUEUE:.*]] = arith.constant 1089156 : i32
// CHECK: aiex.npu.write32(%[[QUEUE]], %{{.*}}) : i32, i32
module @mm2s4_bd15 {
  aie.device(npu3) {
    %mem = aie.tile(0, 1)
    %buf = aie.buffer(%mem) : memref<64xi32>
    aie.runtime_sequence() {
      %task = aiex.dma_configure_task(%mem, MM2S, 4) {
        aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64) {bd_id = 15 : i32}
        aie.end
      }
      aiex.dma_start_task(%task)
    }
  }
}

// -----

// CHECK-LABEL: @mm2s5_bd0
// CHECK: aiex.npu.blockwrite
// CHECK-SAME: address = 1346560 : ui32
// CHECK: %[[QUEUE:.*]] = arith.constant 1351268 : i32
// CHECK: aiex.npu.write32(%[[QUEUE]], %{{.*}}) : i32, i32
module @mm2s5_bd0 {
  aie.device(npu3) {
    %mem = aie.tile(0, 1)
    %buf = aie.buffer(%mem) : memref<64xi32>
    aie.runtime_sequence() {
      %task = aiex.dma_configure_task(%mem, MM2S, 5) {
        aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%task)
    }
  }
}

// -----

// CHECK-LABEL: @mm2s9_bd15
// CHECK: aiex.npu.blockwrite
// CHECK-SAME: address = 1350352 : ui32
// CHECK: %[[QUEUE:.*]] = arith.constant 1351300 : i32
// CHECK: aiex.npu.write32(%[[QUEUE]], %{{.*}}) : i32, i32
module @mm2s9_bd15 {
  aie.device(npu3) {
    %mem = aie.tile(0, 1)
    %buf = aie.buffer(%mem) : memref<64xi32>
    aie.runtime_sequence() {
      %task = aiex.dma_configure_task(%mem, MM2S, 9) {
        aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64) {bd_id = 15 : i32}
        aie.end
      }
      aiex.dma_start_task(%task)
    }
  }
}

// -----

// CHECK-LABEL: @s2mm1_bd0
// CHECK: aiex.npu.blockwrite
// CHECK-SAME: address = 1082112 : ui32
// CHECK: %[[QUEUE:.*]] = arith.constant 1089036 : i32
// CHECK: aiex.npu.write32(%[[QUEUE]], %{{.*}}) : i32, i32
module @s2mm1_bd0 {
  aie.device(npu3) {
    %mem = aie.tile(0, 1)
    %buf = aie.buffer(%mem) : memref<64xi32>
    aie.runtime_sequence() {
      %task = aiex.dma_configure_task(%mem, S2MM, 1) {
        aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%task)
    }
  }
}

// -----

// CHECK-LABEL: @s2mm4_bd15
// CHECK: aiex.npu.blockwrite
// CHECK-SAME: address = 1344208 : ui32
// CHECK: %[[QUEUE:.*]] = arith.constant 1351172 : i32
// CHECK: aiex.npu.write32(%[[QUEUE]], %{{.*}}) : i32, i32
module @s2mm4_bd15 {
  aie.device(npu3) {
    %mem = aie.tile(0, 1)
    %buf = aie.buffer(%mem) : memref<64xi32>
    aie.runtime_sequence() {
      %task = aiex.dma_configure_task(%mem, S2MM, 4) {
        aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64) {bd_id = 15 : i32}
        aie.end
      }
      aiex.dma_start_task(%task)
    }
  }
}

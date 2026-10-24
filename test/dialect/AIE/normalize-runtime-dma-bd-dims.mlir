// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-normalize-dma-bd-dims --split-input-file %s | FileCheck %s
// RUN: aie-opt --aie-normalize-dma-bd-dims --aie-materialize-bd-chains --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu --split-input-file %s | FileCheck %s --check-prefix=LOWER

// Runtime layouts reserve the outermost entry for iteration when there are
// four entries, and for [iteration, D3] when there are five. Removing even a
// singleton entry can change the interpretation of every remaining dimension.
// Check concrete and symbolic tasks, including singleton addressing entries.

// CHECK-LABEL: @runtime_d3
// CHECK: sizes = [1, 2, 2, 3, 4] strides = [0, 30, 12, 4, 1]
// CHECK: sizes = [1, 1, 2, 3, 4] strides = [0, 30, 16, 4, 1]
// CHECK: sizes = [1, 2, 1, 3, 4] strides = [0, 30, 12, 4, 1]
// CHECK: sizes = [1, 2, 2, 1, 4] strides = [0, 30, 12, 4, 1]
// CHECK: sizes = [1, 2, 2, 3, 1] strides = [0, 30, 12, 4, 1]
// LOWER-LABEL: @runtime_d3
// LOWER: buffer_length = 48 : i32
// LOWER-SAME: d2_size = 2 : i32
// LOWER-SAME: d3_size = 2 : i32, d3_stride = 30 : i32
// LOWER: buffer_length = 24 : i32
// LOWER-SAME: d2_size = 2 : i32
// LOWER-SAME: d3_size = 1 : i32, direction
// LOWER: buffer_length = 24 : i32
// LOWER-SAME: d2_size = 1 : i32
// LOWER-SAME: d3_size = 2 : i32, d3_stride = 30 : i32
// LOWER: buffer_length = 16 : i32
// LOWER-SAME: d1_size = 1 : i32
// LOWER-SAME: d3_size = 2 : i32, d3_stride = 30 : i32
// LOWER: buffer_length = 12 : i32
// LOWER-SAME: d0_size = 1 : i32
// LOWER-SAME: d3_size = 2 : i32, d3_stride = 30 : i32
module @runtime_d3 {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @input (%shim, MM2S, 0)
    aie.runtime_sequence(%arg: memref<128xi32>) {
      %task0 = aiex.dma_configure_task_for @input {
        aie.dma_bd(%arg : memref<128xi32> offset = 0 len = 48 sizes = [1, 2, 2, 3, 4] strides = [0, 30, 12, 4, 1])
        aie.end
      }
      aiex.dma_start_task(%task0)
      %task1 = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%arg : memref<128xi32> offset = 0 len = 24 sizes = [1, 1, 2, 3, 4] strides = [0, 30, 16, 4, 1])
        aie.end
      }
      aiex.dma_start_task(%task1)
      %task2 = aiex.dma_configure_task_for @input {
        aie.dma_bd(%arg : memref<128xi32> offset = 0 len = 24 sizes = [1, 2, 1, 3, 4] strides = [0, 30, 12, 4, 1])
        aie.end
      }
      aiex.dma_start_task(%task2)
      %task3 = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%arg : memref<128xi32> offset = 0 len = 16 sizes = [1, 2, 2, 1, 4] strides = [0, 30, 12, 4, 1])
        aie.end
      }
      aiex.dma_start_task(%task3)
      %task4 = aiex.dma_configure_task_for @input {
        aie.dma_bd(%arg : memref<128xi32> offset = 0 len = 12 sizes = [1, 2, 2, 3, 1] strides = [0, 30, 12, 4, 1])
        aie.end
      }
      aiex.dma_start_task(%task4)
    }
  }
}

// -----

// On AIE4, even a size-one iteration slot in four-entry layouts must remain
// separate from addressing, and contiguous task layouts retain their rank.
// CHECK-LABEL: @runtime_legacy
// CHECK: sizes = [1, 1, 2, 4] strides = [0, 0, 8, 1]
// CHECK: sizes = [3, 1, 1, 8] strides = [8, 0, 0, 1]
// LOWER-LABEL: @runtime_legacy
// LOWER: buffer_length = 8 : i32
// LOWER-SAME: d1_size = 2 : i32, d1_stride = 8 : i32
// LOWER-SAME: iteration_size = 0 : i32
// LOWER: buffer_length = 8 : i32
// LOWER-SAME: iteration_size = 2 : i32
module @runtime_legacy {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.runtime_sequence(%arg: memref<32xi32>) {
      %task0 = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%arg : memref<32xi32> offset = 0 len = 8 sizes = [1, 1, 2, 4] strides = [0, 0, 8, 1])
        aie.end
      }
      aiex.dma_start_task(%task0)
      %task1 = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%arg : memref<32xi32> offset = 0 len = 8 sizes = [3, 1, 1, 8] strides = [8, 0, 0, 1])
        aie.end
      } {repeat_count = 2 : i32}
      aiex.dma_start_task(%task1)
    }
  }
}

// -----

// An abstract chain is outside a runtime sequence until materialized. Its
// singleton entries must also survive normalization before materialization.
// CHECK-LABEL: @abstract_chain
// CHECK: aie.bd_chain @d3
// CHECK: sizes = [1, 2, 1, 3, 4] strides = [0, 30, 12, 4, 1]
// LOWER-LABEL: @abstract_chain
// LOWER: buffer_length = 24 : i32
// LOWER-SAME: d2_size = 1 : i32
// LOWER-SAME: d3_size = 2 : i32, d3_stride = 30 : i32
module @abstract_chain {
  aie.device(npu3) {
    %shim = aie.tile(0, 0)
    aie.bd_chain @d3(%arg: memref<128xi32>) {
      aie.dma_bd(%arg : memref<128xi32> offset = 0 len = 24 sizes = [1, 2, 1, 3, 4] strides = [0, 30, 12, 4, 1])
      aie.end
    }
    aie.runtime_sequence(%arg: memref<128xi32>) {
      %task = aiex.dma_start_bd_chain @d3(%arg) : (memref<128xi32>) on (%shim, MM2S, 0)
    }
  }
}

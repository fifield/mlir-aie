//===- npu3_bd_layout_errors.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-dma-to-npu %s 2>&1 | FileCheck %s

// AIE4 memtile BD ids are per channel pool, so writebd needs a channel.
// CHECK: error: 'aiex.npu.writebd' op needs a channel and direction: a BD id on this tile names a slot of its channel's BD pool

// Shim MM2S channel 2 uses the upper bank, ids 16-31.
// CHECK: error: 'aiex.npu.dma_memcpy_nd' op programs buffer descriptor ID 3, which is not in the BD pool of MM2S channel 2 of tile (0, 0): ids 16 to 31

aie.device(npu3) {
  %m11 = aie.tile(1, 1)
  aie.runtime_sequence @writebd_no_channel() {
    aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 256 : i32, buffer_offset = 0 : i32, column = 1 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 1 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
  }
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @in2(%shim, MM2S, 2)
  aie.runtime_sequence @memcpy_wrong_bank(%arg0 : memref<64xi32>) {
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 3 : i64, metadata = @in2} : memref<64xi32>
  }
}

// -----

// CHECK: error: 'aiex.npu.writebd' op channel and direction must be given together

aie.device(npu3) {
  %m11 = aie.tile(1, 1)
  aie.runtime_sequence @writebd_channel_only() {
    aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 256 : i32, buffer_offset = 0 : i32, channel = 0 : i32, column = 1 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 1 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
  }
}

// -----

// The AIE4 memtile pad fields are 8 bits wide.
// CHECK: error: 'aiex.npu.writebd' op D0 pad_before/pad_after exceeds the [0:255] range.

aie.device(npu3) {
  %m11 = aie.tile(1, 1)
  aie.runtime_sequence @writebd_pad() {
    aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 256 : i32, buffer_offset = 0 : i32, channel = 0 : i32, column = 1 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 256 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, direction = 1 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 1 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
  }
}

// -----

// CHECK: error: 'aiex.npu.dma_memcpy_nd' op Stride 0 is 2, but must be 1: this tile's buffer descriptors have no d0 stride, the innermost dimension is always contiguous.

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @in0(%shim, MM2S, 0)
  aie.runtime_sequence @memcpy_d0_stride(%arg0 : memref<256xi32>) {
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 8, 16][0, 0, 32, 2]) {id = 0 : i64, metadata = @in0} : memref<256xi32>
  }
}

//===- bad_npu_write_bd_d0_stride_npu3.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// AIE4 BDs have no d0 stride field, so with d0_size > 1 only d0_stride 0 or 1
// (unbiased granules) is representable.

aie.device(npu3) {
  %mem = aie.tile(0, 1)
  aie.runtime_sequence @seq() {
    // expected-error@+1 {{'aiex.npu.writebd' op D0 Stride must encode a unit step: the BDs of this tile have no d0 stride field.}}
    aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 256 : i32, buffer_offset = 0 : i32, channel = 0 : i32, column = 0 : i32, d0_size = 16 : i32, d0_stride = 2 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 16 : i32, d1_stride = 16 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, direction = 1 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 1 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
  }
}

// -----

aie.device(npu3) {
  %core = aie.tile(0, 2)
  aie.runtime_sequence @seq() {
    // expected-error@+1 {{'aiex.npu.writebd' op D0 Stride must encode a unit step: the BDs of this tile have no d0 stride field.}}
    aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 256 : i32, buffer_offset = 0 : i32, channel = 0 : i32, column = 0 : i32, d0_size = 16 : i32, d0_stride = 3 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 16 : i32, d1_stride = 16 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, direction = 1 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 2 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
  }
}

// -----

// A unit step is accepted, and so is any d0_stride when d0_size is 1.

aie.device(npu3) {
  %mem = aie.tile(0, 1)
  aie.runtime_sequence @seq() {
    aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 256 : i32, buffer_offset = 0 : i32, channel = 0 : i32, column = 0 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 16 : i32, d1_stride = 16 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, direction = 1 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 1 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
    aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 256 : i32, buffer_offset = 0 : i32, channel = 0 : i32, column = 0 : i32, d0_size = 16 : i32, d0_stride = 1 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 16 : i32, d1_stride = 16 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, direction = 1 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 1 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
    aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 256 : i32, buffer_offset = 0 : i32, channel = 0 : i32, column = 0 : i32, d0_size = 1 : i32, d0_stride = 2 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 16 : i32, d1_stride = 16 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, direction = 1 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 1 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
  }
}

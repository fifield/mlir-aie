//===- bad_npu_write_bd_aie4.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// Verify that NpuWriteBdOp on AIE4 (npu3z) correctly enforces 12-bit
// dimension size limits (max 4095) instead of the AIE2 10-bit limits (1023).

// AIE4 shim: D0 size overflow (4096 > 4095)
module {
  aie.device(npu3z) {
    aie.runtime_sequence(%in : memref<131072xi32>, %out : memref<131072xi32>) {
      // expected-error@+1 {{D0 Size exceeds the [0:4095] range.}}
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 32 : i32, buffer_offset = 0 : i32, channel = 0 : i32, column = 0 : i32, row = 0 : i32, d0_stride = 0 : i32, d0_size = 4096 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_stride = 7 : i32, d1_size = 2 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, direction = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stride = 0 : i32, iteration_size = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
    }
  }
}

// -----

// AIE4 shim: D1 size overflow (4096 > 4095)
module {
  aie.device(npu3z) {
    aie.runtime_sequence(%in : memref<131072xi32>, %out : memref<131072xi32>) {
      // expected-error@+1 {{D1 Size exceeds the [0:4095] range.}}
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 32 : i32, buffer_offset = 0 : i32, channel = 0 : i32, column = 0 : i32, row = 0 : i32, d0_stride = 0 : i32, d0_size = 8 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_stride = 7 : i32, d1_size = 4096 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, direction = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stride = 0 : i32, iteration_size = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
    }
  }
}

// -----

// AIE4 shim: D0 size=4095 should be accepted (boundary)
// AIE4 shim: D1 size=4095 should be accepted (boundary)
// AIE4 shim: D2 size allowed (unlike AIE2 which rejects D2 on shim)
module {
  aie.device(npu3z) {
    aie.runtime_sequence(%in : memref<131072xi32>, %out : memref<131072xi32>) {
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 32 : i32, buffer_offset = 0 : i32, channel = 0 : i32, column = 0 : i32, row = 0 : i32, d0_stride = 0 : i32, d0_size = 4095 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_stride = 7 : i32, d1_size = 4095 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 8 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, direction = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stride = 0 : i32, iteration_size = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
    }
  }
}

// -----

// AIE2 shim: D0 size=1024 should still be rejected (AIE2 max is 1023)
// Verify AIE2 behavior is not impacted.
module {
  aie.device(npu1) {
    aie.runtime_sequence(%in : memref<131072xi32>, %out : memref<131072xi32>) {
      // expected-error@+1 {{D0 Size exceeds the [0:1023] range.}}
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 32 : i32, buffer_offset = 0 : i32, channel = 0 : i32, column = 0 : i32, row = 0 : i32, d0_stride = 0 : i32, d0_size = 1024 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_stride = 7 : i32, d1_size = 2 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, direction = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stride = 0 : i32, iteration_size = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
    }
  }
}

// -----

// AIE2 shim: D2 size still rejected on AIE2
module {
  aie.device(npu1) {
    aie.runtime_sequence(%in : memref<131072xi32>, %out : memref<131072xi32>) {
      // expected-error@+1 {{ShimTile only supports 3 dimensions of sizes.}}
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 32 : i32, buffer_offset = 0 : i32, channel = 0 : i32, column = 0 : i32, row = 0 : i32, d0_stride = 0 : i32, d0_size = 8 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_stride = 7 : i32, d1_size = 2 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 4 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, direction = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stride = 0 : i32, iteration_size = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
    }
  }
}

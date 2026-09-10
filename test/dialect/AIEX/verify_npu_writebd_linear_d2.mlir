//===- verify_npu_writebd_linear_d2.mlir ----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// Positive coverage for the NpuWriteBdOp "linear transfer" classification.
//
// A BD is only linear (exempt from the D0 wrap-size limit, because the wide
// buffer_length register carries the length) when NO higher addressing
// dimension is active. The check must consider d2_size: an active D2
// (d2_size > 1) means the BD uses ND wrap fields, so the D0 size must fit the
// hardware D0_Wrap field (10 bits / 1023 on AIE2, 12 bits / 4095 on AIE4 for
// shim/MemTile BDs -- per the register database). A previous version of the
// predicate omitted d2_size, so a BD with d1_size==1 but d2_size>1 was wrongly
// treated as linear and skipped the D0 range check.

// -----
// AIE4 shim: d1_size==1 but d2_size>1 (active D2) -> NOT linear -> D0 size
// (4096) must be rejected against the 4095 limit.
module {
  aie.device(npu3) {
    aie.runtime_sequence(%in : memref<131072xi32>, %out : memref<131072xi32>) {
      // expected-error@+1 {{D0 Size exceeds the [0:4095] range.}}
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 32 : i32, buffer_offset = 0 : i32, channel = 0 : i32, column = 0 : i32, row = 0 : i32, d0_stride = 0 : i32, d0_size = 4096 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_stride = 0 : i32, d1_size = 1 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 2 : i32, d2_stride = 4 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, d3_size = 1 : i32, d3_stride = 0 : i32, direction = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stride = 0 : i32, iteration_size = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
    }
  }
}

// -----
// AIE4 shim: genuinely linear (d1==d2==d3==1, iteration==0) -> the D0 wrap-size
// limit is exempt, so a large D0 (4096) is accepted (no diagnostic). This
// confirms adding the d2_size term did not break the linear exemption.
module {
  aie.device(npu3) {
    aie.runtime_sequence(%in : memref<131072xi32>, %out : memref<131072xi32>) {
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 4096 : i32, buffer_offset = 0 : i32, channel = 0 : i32, column = 0 : i32, row = 0 : i32, d0_stride = 0 : i32, d0_size = 4096 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_stride = 0 : i32, d1_size = 1 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 1 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, d3_size = 1 : i32, d3_stride = 0 : i32, direction = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stride = 0 : i32, iteration_size = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
    }
  }
}

// -----
// AIE2 shim: d1_size==1 but d2_size>1 (active D2) -> NOT linear -> D0 size
// (1024) must be rejected against the 1023 limit.
module {
  aie.device(npu1) {
    aie.runtime_sequence(%in : memref<131072xi32>, %out : memref<131072xi32>) {
      // expected-error@+1 {{D0 Size exceeds the [0:1023] range.}}
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 32 : i32, buffer_offset = 0 : i32, channel = 0 : i32, column = 0 : i32, row = 0 : i32, d0_stride = 0 : i32, d0_size = 1024 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_stride = 0 : i32, d1_size = 1 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 2 : i32, d2_stride = 4 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, direction = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stride = 0 : i32, iteration_size = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
    }
  }
}

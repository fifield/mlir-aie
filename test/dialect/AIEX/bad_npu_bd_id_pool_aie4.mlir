//===- bad_npu_bd_id_pool_aie4.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// npu.writebd and npu.push_queue both carry a direction and a channel, so the
// bd_id bound is the target channel's own BD pool rather than the flat
// per-tile count. direction 2 = ControlMM2S (8 private BDs), 3 = TraceS2MM (4).

// ControlMM2S bd 8 is past the end of the 8-entry control bank.
aie.device(npu3) {
  aie.runtime_sequence(%in : memref<8xi32>) {
    // expected-error@+1 {{BD ID 8 is not accessible from channel (ControlMM2S, 0)}}
    aiex.npu.writebd {bd_id = 8 : i32, buffer_length = 8 : i32, buffer_offset = 0 : i32, channel = 0 : i32, column = 0 : i32, row = 0 : i32, d0_stride = 0 : i32, d0_size = 8 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_stride = 0 : i32, d1_size = 1 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, direction = 2 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stride = 0 : i32, iteration_size = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
  }
}

// -----

// TraceS2MM bd 4 is past the end of the 4-entry trace bank.
aie.device(npu3) {
  aie.runtime_sequence(%in : memref<8xi32>) {
    // expected-error@+1 {{BD ID 4 is not accessible from channel (TraceS2MM, 0)}}
    aiex.npu.writebd {bd_id = 4 : i32, buffer_length = 8 : i32, buffer_offset = 0 : i32, channel = 0 : i32, column = 0 : i32, row = 0 : i32, d0_stride = 0 : i32, d0_size = 8 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_stride = 0 : i32, d1_size = 1 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, direction = 3 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stride = 0 : i32, iteration_size = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
  }
}

// -----

// Same bound on the queue push.
aie.device(npu3) {
  aie.runtime_sequence(%in : memref<8xi32>) {
    %c8 = arith.constant 8 : i32
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{BD ID 8 is not accessible from channel (ControlMM2S, 0)}}
    aiex.npu.push_queue(0, 0, ControlMM2S : 0) bd_id %c8 repeat %c0 {issue_token = true} : i32, i32
  }
}

// -----

// The AIE4b shim's upper MM2S pool spans ids 16..31, so a bd_id *below* 16 on
// channel 2 is out of pool. getDmaBdAddress converts to a bank-local index
// with an unguarded `bd_id -= 16`, so 5 would underflow uint32_t and produce
// an address roughly 4GiB from the intended one, with no diagnostic. This is a
// plain data channel: the bound matters well beyond the specialized ones.
aie.device(npu3) {
  aie.runtime_sequence(%in : memref<8xi32>) {
    // expected-error@+1 {{BD ID 5 is not accessible from channel (MM2S, 2)}}
    aiex.npu.writebd {bd_id = 5 : i32, buffer_length = 8 : i32, buffer_offset = 0 : i32, channel = 2 : i32, column = 0 : i32, row = 0 : i32, d0_stride = 0 : i32, d0_size = 8 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_stride = 0 : i32, d1_size = 1 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, direction = 1 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stride = 0 : i32, iteration_size = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
  }
}

// -----

// An out-of-range direction must be rejected rather than falling through
// getBdPoolId's if-chain and being silently treated as S2MM.
aie.device(npu3) {
  aie.runtime_sequence(%in : memref<8xi32>) {
    // expected-error@+1 {{invalid direction 7}}
    aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 8 : i32, buffer_offset = 0 : i32, channel = 0 : i32, column = 0 : i32, row = 0 : i32, d0_stride = 0 : i32, d0_size = 8 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_stride = 0 : i32, d1_size = 1 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, direction = 7 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stride = 0 : i32, iteration_size = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
  }
}

// -----

// The AIE4b shim's upper MM2S pool spans ids 16..31, so bd 20 on channel 2 is
// legal even though the pool holds only 16 entries. This is the case a
// size-based bound (rather than a range-based one) would wrongly reject.
aie.device(npu3) {
  aie.runtime_sequence(%in : memref<8xi32>) {
    aiex.npu.writebd {bd_id = 20 : i32, buffer_length = 8 : i32, buffer_offset = 0 : i32, channel = 2 : i32, column = 0 : i32, row = 0 : i32, d0_stride = 0 : i32, d0_size = 8 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_stride = 0 : i32, d1_size = 1 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, direction = 1 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stride = 0 : i32, iteration_size = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
  }
}

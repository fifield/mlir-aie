// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// Unsupported DMA engines and channels must be diagnosed before BD pool queries.

// Trace DMA does not exist on NPU2.
aie.device(npu2) {
  aie.runtime_sequence() {
    // expected-error@+1 {{unsupported DMA channel (TraceS2MM, 0) on tile (0, 0)}}
    aiex.npu.writebd {
      column = 0 : i32, row = 0 : i32,
      direction = 3 : i32, channel = 0 : i32, bd_id = 3 : i32,
      buffer_length = 8 : i32, buffer_offset = 0 : i32,
      enable_packet = 0 : i32, out_of_order_id = 0 : i32,
      packet_id = 0 : i32, packet_type = 0 : i32,
      d0_size = 8 : i32, d0_stride = 0 : i32,
      d1_size = 1 : i32, d1_stride = 0 : i32,
      d2_size = 0 : i32, d2_stride = 0 : i32,
      iteration_current = 0 : i32, iteration_size = 0 : i32,
      iteration_stride = 0 : i32,
      next_bd = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32,
      lock_rel_val = 0 : i32, lock_rel_id = 0 : i32,
      lock_acq_enable = 0 : i32, lock_acq_val = 0 : i32, lock_acq_id = 0 : i32,
      d0_zero_before = 0 : i32, d0_zero_after = 0 : i32,
      d1_zero_before = 0 : i32, d1_zero_after = 0 : i32,
      d2_zero_before = 0 : i32, d2_zero_after = 0 : i32
    }
  }
}

// -----

// Control DMA is only available on AIE4 shim NOC tiles.
aie.device(npu3) {
  aie.runtime_sequence() {
    // expected-error@+1 {{unsupported DMA channel (ControlMM2S, 0) on tile (0, 1)}}
    aiex.npu.writebd {
      column = 0 : i32, row = 1 : i32,
      direction = 2 : i32, channel = 0 : i32, bd_id = 3 : i32,
      buffer_length = 8 : i32, buffer_offset = 0 : i32,
      enable_packet = 0 : i32, out_of_order_id = 0 : i32,
      packet_id = 0 : i32, packet_type = 0 : i32,
      d0_size = 8 : i32, d0_stride = 0 : i32,
      d1_size = 1 : i32, d1_stride = 0 : i32,
      d2_size = 0 : i32, d2_stride = 0 : i32,
      iteration_current = 0 : i32, iteration_size = 0 : i32,
      iteration_stride = 0 : i32,
      next_bd = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32,
      lock_rel_val = 0 : i32, lock_rel_id = 0 : i32,
      lock_acq_enable = 0 : i32, lock_acq_val = 0 : i32, lock_acq_id = 0 : i32,
      d0_zero_before = 0 : i32, d0_zero_after = 0 : i32,
      d1_zero_before = 0 : i32, d1_zero_after = 0 : i32,
      d2_zero_before = 0 : i32, d2_zero_after = 0 : i32
    }
  }
}

// -----

// Trace DMA has only one channel.
aie.device(npu3) {
  aie.runtime_sequence() {
    // expected-error@+1 {{unsupported DMA channel (TraceS2MM, 1) on tile (0, 0)}}
    aiex.npu.writebd {
      column = 0 : i32, row = 0 : i32,
      direction = 3 : i32, channel = 1 : i32, bd_id = 3 : i32,
      buffer_length = 8 : i32, buffer_offset = 0 : i32,
      enable_packet = 0 : i32, out_of_order_id = 0 : i32,
      packet_id = 0 : i32, packet_type = 0 : i32,
      d0_size = 8 : i32, d0_stride = 0 : i32,
      d1_size = 1 : i32, d1_stride = 0 : i32,
      d2_size = 0 : i32, d2_stride = 0 : i32,
      iteration_current = 0 : i32, iteration_size = 0 : i32,
      iteration_stride = 0 : i32,
      next_bd = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32,
      lock_rel_val = 0 : i32, lock_rel_id = 0 : i32,
      lock_acq_enable = 0 : i32, lock_acq_val = 0 : i32, lock_acq_id = 0 : i32,
      d0_zero_before = 0 : i32, d0_zero_after = 0 : i32,
      d1_zero_before = 0 : i32, d1_zero_after = 0 : i32,
      d2_zero_before = 0 : i32, d2_zero_after = 0 : i32
    }
  }
}

// -----

// Ordinary DMA channels are bounded too.
aie.device(npu3) {
  aie.runtime_sequence() {
    // expected-error@+1 {{unsupported DMA channel (MM2S, 4) on tile (0, 0)}}
    aiex.npu.writebd {
      column = 0 : i32, row = 0 : i32,
      direction = 1 : i32, channel = 4 : i32, bd_id = 3 : i32,
      buffer_length = 8 : i32, buffer_offset = 0 : i32,
      enable_packet = 0 : i32, out_of_order_id = 0 : i32,
      packet_id = 0 : i32, packet_type = 0 : i32,
      d0_size = 8 : i32, d0_stride = 0 : i32,
      d1_size = 1 : i32, d1_stride = 0 : i32,
      d2_size = 0 : i32, d2_stride = 0 : i32,
      iteration_current = 0 : i32, iteration_size = 0 : i32,
      iteration_stride = 0 : i32,
      next_bd = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32,
      lock_rel_val = 0 : i32, lock_rel_id = 0 : i32,
      lock_acq_enable = 0 : i32, lock_acq_val = 0 : i32, lock_acq_id = 0 : i32,
      d0_zero_before = 0 : i32, d0_zero_after = 0 : i32,
      d1_zero_before = 0 : i32, d1_zero_after = 0 : i32,
      d2_zero_before = 0 : i32, d2_zero_after = 0 : i32
    }
  }
}

// -----

// The supported trace channel and its last BD remain valid.
aie.device(npu3) {
  aie.runtime_sequence() {
    aiex.npu.writebd {
      column = 0 : i32, row = 0 : i32,
      direction = 3 : i32, channel = 0 : i32, bd_id = 3 : i32,
      buffer_length = 8 : i32, buffer_offset = 0 : i32,
      enable_packet = 0 : i32, out_of_order_id = 0 : i32,
      packet_id = 0 : i32, packet_type = 0 : i32,
      d0_size = 8 : i32, d0_stride = 0 : i32,
      d1_size = 1 : i32, d1_stride = 0 : i32,
      d2_size = 0 : i32, d2_stride = 0 : i32,
      iteration_current = 0 : i32, iteration_size = 0 : i32,
      iteration_stride = 0 : i32,
      next_bd = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32,
      lock_rel_val = 0 : i32, lock_rel_id = 0 : i32,
      lock_acq_enable = 0 : i32, lock_acq_val = 0 : i32, lock_acq_id = 0 : i32,
      d0_zero_before = 0 : i32, d0_zero_after = 0 : i32,
      d1_zero_before = 0 : i32, d1_zero_after = 0 : i32,
      d2_zero_before = 0 : i32, d2_zero_after = 0 : i32
    }
  }
}

// -----

// The second control channel remains valid.
aie.device(npu3) {
  aie.runtime_sequence() {
    aiex.npu.writebd {
      column = 0 : i32, row = 0 : i32,
      direction = 2 : i32, channel = 1 : i32, bd_id = 3 : i32,
      buffer_length = 8 : i32, buffer_offset = 0 : i32,
      enable_packet = 0 : i32, out_of_order_id = 0 : i32,
      packet_id = 0 : i32, packet_type = 0 : i32,
      d0_size = 8 : i32, d0_stride = 0 : i32,
      d1_size = 1 : i32, d1_stride = 0 : i32,
      d2_size = 0 : i32, d2_stride = 0 : i32,
      iteration_current = 0 : i32, iteration_size = 0 : i32,
      iteration_stride = 0 : i32,
      next_bd = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32,
      lock_rel_val = 0 : i32, lock_rel_id = 0 : i32,
      lock_acq_enable = 0 : i32, lock_acq_val = 0 : i32, lock_acq_id = 0 : i32,
      d0_zero_before = 0 : i32, d0_zero_after = 0 : i32,
      d1_zero_before = 0 : i32, d1_zero_after = 0 : i32,
      d2_zero_before = 0 : i32, d2_zero_after = 0 : i32
    }
  }
}

// -----

// Reject specialized engines on older targets for queue pushes too.
aie.device(npu2) {
  aie.runtime_sequence() {
    %bd = arith.constant 0 : i32
    %repeat = arith.constant 0 : i32
    // expected-error@+1 {{unsupported DMA channel (ControlMM2S, 0) on tile (0, 0)}}
    aiex.npu.push_queue(0, 0, ControlMM2S : 0) bd_id %bd repeat %repeat {issue_token = true} : i32, i32
  }
}

// -----

// Core tiles have no trace DMA engine.
aie.device(npu3) {
  aie.runtime_sequence() {
    %bd = arith.constant 0 : i32
    %repeat = arith.constant 0 : i32
    // expected-error@+1 {{unsupported DMA channel (TraceS2MM, 0) on tile (0, 2)}}
    aiex.npu.push_queue(0, 2, TraceS2MM : 0) bd_id %bd repeat %repeat {issue_token = true} : i32, i32
  }
}

// -----

// A runtime BD ID must not bypass channel validation.
aie.device(npu3) {
  aie.runtime_sequence(%bd : i32) {
    %repeat = arith.constant 0 : i32
    // expected-error@+1 {{unsupported DMA channel (TraceS2MM, 1) on tile (0, 0)}}
    aiex.npu.push_queue(0, 0, TraceS2MM : 1) bd_id %bd repeat %repeat {issue_token = true} : i32, i32
  }
}

// -----

// Control DMA has only two channels.
aie.device(npu3) {
  aie.runtime_sequence() {
    %bd = arith.constant 0 : i32
    %repeat = arith.constant 0 : i32
    // expected-error@+1 {{unsupported DMA channel (ControlMM2S, 2) on tile (0, 0)}}
    aiex.npu.push_queue(0, 0, ControlMM2S : 2) bd_id %bd repeat %repeat {issue_token = true} : i32, i32
  }
}

// -----

// Reject negative channel indices before pool lookup.
aie.device(npu3) {
  aie.runtime_sequence() {
    %bd = arith.constant 0 : i32
    %repeat = arith.constant 0 : i32
    // expected-error@+1 {{unsupported DMA channel (S2MM, -1) on tile (0, 0)}}
    aiex.npu.push_queue(0, 0, S2MM : -1) bd_id %bd repeat %repeat {issue_token = true} : i32, i32
  }
}

// -----

// Reject coordinates outside the target.
aie.device(npu3) {
  aie.runtime_sequence() {
    %bd = arith.constant 0 : i32
    %repeat = arith.constant 0 : i32
    // expected-error@+1 {{unsupported DMA channel (MM2S, 0) on tile (0, 99)}}
    aiex.npu.push_queue(0, 99, MM2S : 0) bd_id %bd repeat %repeat {issue_token = true} : i32, i32
  }
}

// -----

// PL-only shims have no memory DMA engine.
aie.device(xcvc1902) {
  aie.runtime_sequence() {
    %bd = arith.constant 0 : i32
    %repeat = arith.constant 0 : i32
    // expected-error@+1 {{unsupported DMA channel (MM2S, 0) on tile (0, 0)}}
    aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %bd repeat %repeat {issue_token = true} : i32, i32
  }
}

// -----

// Runtime BD IDs remain valid on supported channels.
aie.device(npu3) {
  aie.runtime_sequence(%bd : i32) {
    %repeat = arith.constant 0 : i32
    aiex.npu.push_queue(0, 0, TraceS2MM : 0) bd_id %bd repeat %repeat {issue_token = true} : i32, i32
    aiex.npu.push_queue(0, 0, ControlMM2S : 1) bd_id %bd repeat %repeat {issue_token = true} : i32, i32
    aiex.npu.push_queue(0, 0, MM2S : 2) bd_id %bd repeat %repeat {issue_token = true} : i32, i32
  }
}

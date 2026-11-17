//===- bad_npu_push_queue_channel_npu3.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// An AIE4 shim has MM2S channels 0-3 and S2MM channels 0-1.

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.runtime_sequence @seq() {
    %bd = arith.constant 16 : i32
    %rc = arith.constant 0 : i32
    // expected-error@+1 {{'aiex.npu.push_queue' op MM2S channel 4 is out of range: the tile has 4 MM2S channels}}
    aiex.npu.push_queue (0, 0, MM2S:4) bd_id %bd repeat %rc {issue_token = true} : i32, i32
  }
}

// -----

aie.device(npu3) {
  %shim = aie.tile(0, 0)
  aie.runtime_sequence @seq(%bd : i32) {
    %rc = arith.constant 0 : i32
    // expected-error@+1 {{'aiex.npu.push_queue' op S2MM channel 2 is out of range: the tile has 2 S2MM channels}}
    aiex.npu.push_queue (0, 0, S2MM:2) bd_id %bd repeat %rc {issue_token = true} : i32, i32
  }
}

// -----

// An AIE4 core tile has one MM2S channel, a memtile MM2S channels 0-9.

aie.device(npu3) {
  %core = aie.tile(0, 2)
  aie.runtime_sequence @seq() {
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{'aiex.npu.push_queue' op MM2S channel 1 is out of range: the tile has 1 MM2S channels}}
    aiex.npu.push_queue (0, 2, MM2S:1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
}

// -----

aie.device(npu3) {
  %mem = aie.tile(0, 1)
  aie.runtime_sequence @seq() {
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{'aiex.npu.push_queue' op MM2S channel 10 is out of range: the tile has 10 MM2S channels}}
    aiex.npu.push_queue (0, 1, MM2S:10) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
}

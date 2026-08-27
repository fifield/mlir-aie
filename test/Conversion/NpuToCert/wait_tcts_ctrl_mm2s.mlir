//===- wait_tcts_ctrl_mm2s.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-npu-to-cert -split-input-file %s | FileCheck %s

// Task-complete-token actor IDs for the AIE4 shim. The shim control-MM2S and
// trace-S2MM channels are NOT part of the regular MM2S/S2MM actor ranges:
// Ctrl_MM2S0/1 are actors 16/17 and Trace_S2MM is actor 1. Getting this wrong
// makes the cert job wait on a channel the design never uses, which deadlocks
// on hardware.
//
// aiex.npu.sync operands are (column, row, direction, channel, cols, rows) and
// DMAChannelDir is S2MM=0, MM2S=1, ControlMM2S=2, TraceS2MM=3. The emitted
// tile_id is (column << 5) | row.

// ControlMM2S channel 0 on shim (0,0) -> actor 16.
// CHECK: aiex.cert.wait_tcts(0, 16, 1)
aie.device(npu3) {
  aie.runtime_sequence @ctrl_mm2s_chan0() {
    %col = arith.constant 0 : i32
    %row = arith.constant 0 : i32
    %dir = arith.constant 2 : i32
    %chan = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    aiex.npu.sync(%col, %row, %dir, %chan, %one, %one) : i32, i32, i32, i32, i32, i32
  }
}

// -----

// ControlMM2S channel 1 -> actor 17.
// CHECK: aiex.cert.wait_tcts(0, 17, 1)
aie.device(npu3) {
  aie.runtime_sequence @ctrl_mm2s_chan1() {
    %col = arith.constant 0 : i32
    %row = arith.constant 0 : i32
    %dir = arith.constant 2 : i32
    %chan = arith.constant 1 : i32
    %one = arith.constant 1 : i32
    aiex.npu.sync(%col, %row, %dir, %chan, %one, %one) : i32, i32, i32, i32, i32, i32
  }
}

// -----

// TraceS2MM channel 0 -> actor 1 (not S2MM0's actor 0).
// CHECK: aiex.cert.wait_tcts(0, 1, 1)
aie.device(npu3) {
  aie.runtime_sequence @trace_s2mm() {
    %col = arith.constant 0 : i32
    %row = arith.constant 0 : i32
    %dir = arith.constant 3 : i32
    %chan = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    aiex.npu.sync(%col, %row, %dir, %chan, %one, %one) : i32, i32, i32, i32, i32, i32
  }
}

// -----

// Regression guard: plain MM2S channel 0 keeps actor 6. ControlMM2S must not
// collapse onto this, which is exactly what happened when the direction fell
// through to the regular shim MM2S table.
// CHECK: aiex.cert.wait_tcts(0, 6, 1)
aie.device(npu3) {
  aie.runtime_sequence @plain_mm2s_unchanged() {
    %col = arith.constant 0 : i32
    %row = arith.constant 0 : i32
    %dir = arith.constant 1 : i32
    %chan = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    aiex.npu.sync(%col, %row, %dir, %chan, %one, %one) : i32, i32, i32, i32, i32, i32
  }
}

// -----

// The column is still encoded into tile_id for a control-MM2S wait.
// CHECK: aiex.cert.wait_tcts(32, 16, 1)
aie.device(npu3) {
  aie.runtime_sequence @ctrl_mm2s_col1() {
    %col = arith.constant 1 : i32
    %row = arith.constant 0 : i32
    %dir = arith.constant 2 : i32
    %chan = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    aiex.npu.sync(%col, %row, %dir, %chan, %one, %one) : i32, i32, i32, i32, i32, i32
  }
}

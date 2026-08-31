//===- wait_tcts_mem_s2mm.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-npu-to-cert -split-input-file %s | FileCheck %s

// AIE4 maps MEM S2MM0 through S2MM7 directly to TCT actors 0 through 7. Cover
// both endpoints so this catches an off-by-one mapping or a missing channel 7.

// MEM tile (0,1), S2MM channel 0 -> tile ID 1, actor 0.
// CHECK: aiex.cert.wait_tcts(1, 0, 1)
aie.device(npu3) {
  aie.runtime_sequence @mem_s2mm_chan0() {
    %col = arith.constant 0 : i32
    %row = arith.constant 1 : i32
    %dir = arith.constant 0 : i32
    %chan = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    aiex.npu.sync(%col, %row, %dir, %chan, %one, %one) : i32, i32, i32, i32, i32, i32
  }
}

// -----

// MEM tile (1,1), S2MM channel 3 -> tile ID 33, actor 3.
// CHECK: aiex.cert.wait_tcts(33, 3, 1)
aie.device(npu3) {
  aie.runtime_sequence @mem_s2mm_chan3_col1() {
    %col = arith.constant 1 : i32
    %row = arith.constant 1 : i32
    %dir = arith.constant 0 : i32
    %chan = arith.constant 3 : i32
    %one = arith.constant 1 : i32
    aiex.npu.sync(%col, %row, %dir, %chan, %one, %one) : i32, i32, i32, i32, i32, i32
  }
}

// -----

// MEM tile (2,1), S2MM channel 7 -> tile ID 65, actor 7.
// CHECK: aiex.cert.wait_tcts(65, 7, 1)
aie.device(npu3) {
  aie.runtime_sequence @mem_s2mm_chan7_col2() {
    %col = arith.constant 2 : i32
    %row = arith.constant 1 : i32
    %dir = arith.constant 0 : i32
    %chan = arith.constant 7 : i32
    %one = arith.constant 1 : i32
    aiex.npu.sync(%col, %row, %dir, %chan, %one, %one) : i32, i32, i32, i32, i32, i32
  }
}

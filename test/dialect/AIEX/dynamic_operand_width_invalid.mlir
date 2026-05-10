//===- dynamic_operand_width_invalid.mlir ---------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu1) {
    aie.runtime_sequence(%addr : i64, %value : i32) {
      // expected-error@+1 {{dynamic address must be a 32-bit integer}}
      aiex.npu.write32(%addr, %value) {address = 0 : ui32, value = 0 : ui32} : i64, i32
    }
  }
}

// -----

module {
  aie.device(npu1) {
    aie.runtime_sequence(%addr : i32, %value : i64) {
      // expected-error@+1 {{dynamic value must be a 32-bit integer}}
      aiex.npu.write32(%addr, %value) {address = 0 : ui32, value = 0 : ui32} : i32, i64
    }
  }
}

// -----

module {
  aie.device(npu1) {
    aie.runtime_sequence(%addr : i32, %value : i64, %mask : i32) {
      // expected-error@+1 {{dynamic operands must be 32-bit integers}}
      aiex.npu.maskwrite32(%addr, %value, %mask) {address = 0 : ui32, mask = 0 : ui32, value = 0 : ui32} : i32, i64, i32
    }
  }
}

// -----

module {
  aie.device(npu1) {
    aie.runtime_sequence(%col : i32, %row : i32, %dir : i32, %chan : i64, %col_num : i32, %row_num : i32) {
      // expected-error@+1 {{all dynamic operands must be 32-bit signless integers}}
      aiex.npu.sync(%col, %row, %dir, %chan, %col_num, %row_num) {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32} : i32, i32, i32, i64, i32, i32
    }
  }
}

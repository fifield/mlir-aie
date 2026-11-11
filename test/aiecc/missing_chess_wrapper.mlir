// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A private install prefix forces the missing-payload case even on Chess CI.
// No Chess installation or license is needed: fail before invoking the linker.
// RUN: %python %S/Inputs/missing_chess_wrapper.py %s %t

module {
  aie.device(npu3) {
    %tile = aie.tile(0, 4)
    aie.core(%tile) {
      aie.end
    }
  }
}

//===- ctrlpkt.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: dont_run
// RUN: echo FAIL | FileCheck %s
// CHECK: PASS

module {
  aie.device(npu3) {
    %shim_0_0 = aie.tile(0, 0)
    %core_0_2 = aie.tile(0, 2)

    aie.runtime_sequence() {
      // col 0 row 2 lock 0
      aiex.control_packet {address = 0x6AE000 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

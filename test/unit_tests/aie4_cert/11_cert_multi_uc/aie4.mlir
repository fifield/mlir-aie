//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
// REQUIRES: dont_run, mesimulator
// RUN: echo FAIL | FileCheck
// CHECK: PASS
//
module {
  aie.device(npu3z) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)

    %tile_0_3 = aie.tile(0, 3)
    %tile_1_2 = aie.tile(1, 2)

    // col 0 row 3 lock 5
    %lock_0_3_5 = aie.lock(%tile_0_3, 5) { init = 3 : i32 }

    // col 1 row 2 lock 2
    %lock_1_2_2 = aie.lock(%tile_1_2, 2) { init = 3 : i32 }

    %core_0_3 = aie.core(%tile_0_3) {
      aie.use_lock(%lock_0_3_5, AcquireGreaterEqual, 5)
      aie.end
    }
    %core_1_2 = aie.core(%tile_1_2) {
      aie.use_lock(%lock_1_2_2, AcquireGreaterEqual, 5)
      aie.end
    }
    // Column 0
    aiex.cert.attach_to_group(0) {
      aiex.cert.job(0) {
        // This barrier will be reached after the implicit configuration job has
        // completed, because they will be on different pages
        aiex.cert.remote_barrier (1, 0x5)
        // col 0 row 3 lock 5
        aiex.npu.write32{ address=0x7AE050 : ui32, value=0x1 : ui32}
      }
    }
    // Column 1
    aiex.cert.attach_to_group(1) {
      aiex.cert.job(1) {
        // Without the barrier, configuration of column 1 locks by column 0
        // will race with the write32
        aiex.cert.remote_barrier (1, 0x5)
        // col 1 row 2 lock 2
        aiex.npu.write32{ address=0x26AE020 : ui32, value=0x4 : ui32}
      }
    }
  }
}

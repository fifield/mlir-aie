//===- aiecc_options_aie4.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// --reconfig-init-entry is rejected for AIE4: it moves each sequence's
// leading reset into an entry run once per hardware context, but the
// partition may be torn down whenever the context is idle, and what that
// resets is not known. --reconfig-teardown alone builds.

// RUN: not %aiecc --get-full-elf --expand-load-pdis --reconfig-teardown --reconfig-init-entry --tmpdir=aiecc_options_aie4_prj/init %s 2>&1 | FileCheck %s
// CHECK: aiecc: --reconfig-init-entry is not supported for AIE4 devices
// RUN: %aiecc --get-full-elf --expand-load-pdis --reconfig-teardown --tmpdir=aiecc_options_aie4_prj/teardown --full-elf-name=aiecc_options_aie4_prj/teardown.elf %s

module {
  aie.device(npu3_1col) @main {
    %s0 = aie.tile(0, 0)
    aie.runtime_sequence @s1(%x: memref<16xi32>) {
      aiex.configure @a {
        aiex.run @run(%x) : (memref<16xi32>)
      }
    }
  }
  aie.device(npu3_1col) @a {
    %t = aie.tile(0, 2)
    %l = aie.lock(%t, 0) {init = 1 : i32}
    aie.runtime_sequence @run(%x: memref<16xi32>) {
    }
  }
}

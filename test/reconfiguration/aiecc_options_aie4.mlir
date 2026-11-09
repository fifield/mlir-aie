//===- aiecc_options_aie4.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// --reconfig-teardown and --reconfig-init-entry are AIE2-only: an AIE4 full ELF
// runs every runtime sequence of the entry device in one cert program, so there
// is no separate once-per-context init entry.

// RUN: not %aiecc --get-full-elf --expand-load-pdis --reconfig-teardown --tmpdir=aiecc_options_aie4_prj/teardown %s 2>&1 | FileCheck %s
// RUN: not %aiecc --get-full-elf --expand-load-pdis --reconfig-teardown --reconfig-init-entry --tmpdir=aiecc_options_aie4_prj/init %s 2>&1 | FileCheck %s
// CHECK: aiecc: --reconfig-teardown and --reconfig-init-entry are not supported for AIE4 devices


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

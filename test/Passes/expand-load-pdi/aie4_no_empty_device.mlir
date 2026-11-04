//===- aie4_no_empty_device.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi %s | FileCheck %s

// The AIE4 firmware does not reset the array on a load, so an empty
// device load ahead of the configuration would reset nothing. Each AIE4 load
// becomes its write_config alone: no @empty_N device, no alternation, and no
// trailing load at the end of the sequence.

// CHECK-NOT: @empty_
// CHECK-LABEL: aie.device(npu3_1col) {
// CHECK: aie.runtime_sequence @seq
// CHECK-NEXT: aiex.npu.write_config @a
// CHECK-NEXT: aiex.npu.write_config @b
// CHECK-NEXT: aiex.npu.write_config @a
// CHECK-NOT: load_pdi
// CHECK-NOT: @empty_

module {
  aie.device(npu3_1col) @a {
    %t = aie.tile(0, 2)
    %l = aie.lock(%t, 0) {init = 1 : i32}
  }
  aie.device(npu3_1col) @b {
    %t = aie.tile(0, 2)
    %l = aie.lock(%t, 0) {init = 2 : i32}
  }
  aie.device(npu3_1col) @main {
    %t = aie.tile(0, 0)
    aie.runtime_sequence @seq(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @a}
      aiex.npu.load_pdi {device_ref = @b}
      aiex.npu.load_pdi {device_ref = @a}
    }
  }
}

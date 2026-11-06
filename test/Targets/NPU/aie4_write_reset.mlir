//===- aie4_write_reset.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aiex.npu.write_reset encodes the AIE4 partition reset into the transaction
// binary. In npu3_1col the last Q0 write, which disables core (0,5) (opcode 3
// MASKWRITE, RegOff 0x9a3000, Value 0x0, Mask 0x1, Size 0x1c), is followed by
// the first core reset readback, a MASKPOLL (opcode 4, RegOff 0x6a3000, Value
// 0x2, Mask 0x3, Size 0x1c).

// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false %s | FileCheck %s

// The reset of core (0,5), then its disable.
// CHECK:      009A3000
// CHECK:      009A3000
// CHECK-NEXT: 00000000
// CHECK-NEXT: 00000000
// CHECK-NEXT: 00000001
// CHECK-NEXT: 0000001C
// CHECK-NEXT: 00000004
// CHECK-NEXT: 00000000
// CHECK-NEXT: 006A3000
// CHECK-NEXT: 00000000
// CHECK-NEXT: 00000002
// CHECK-NEXT: 00000003
// CHECK-NEXT: 0000001C

module {
  aie.device(npu3_1col) @main {
    %t00 = aie.tile(0, 0)
    aie.runtime_sequence @seq(%arg0: memref<1xi32>) {
      aiex.npu.write_reset
    }
  }
}
